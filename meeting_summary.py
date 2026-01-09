# SPDX-FileCopyrightText: 2025 Aurélien Bompard <aurelien@bompard.org>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import re
from contextlib import suppress
from pathlib import Path
from typing import Type, cast
from urllib.parse import urlparse

from aiohttp.web import HTTPError
from google import genai
from google.genai.types import GenerateContentResponse, Part
from maubot import MessageEvent, Plugin  # type: ignore
from maubot.handlers import event
from mautrix import errors
from mautrix.types import (
    CanonicalAliasStateEventContent,
    EventType,
    MessageType,
    RoomAlias,
    RoomID,
)
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

MEETBOT_PREFIX = "Text Log: "
LLM_PROMPT = """
    Give me the key discussion points and action items in this document as a bullet list.
    Do not add an introduction to your response.
    Use markdown formatting.
"""

RESPONSE_TEMPLATE = """Hello {members}. Here's a short summary of the meeting:

---
{summary}
---

Does it look correct to you? If so, please click on ✅, otherwise please click on ❌.

(*this bot feature is in alpha stage, thanks for your patience*)
"""


class MeetingLogFetchingError(Exception):
    pass


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        if "gemini" in self and "model" not in self["gemini"]:
            self["gemini"]["model"] = helper.base["gemini"]["model"]
        helper.copy_dict("gemini")
        helper.copy("meetbot_id")
        helper.copy("ignored_participants")
        helper.copy("meetings_directory")


class MeetingSummary(Plugin):
    config: Config  # pyright: ignore[reportIncompatibleVariableOverride]

    async def start(self) -> None:
        self.config.load_and_update()
        # self.log.debug("Loaded config: %r", dict(self.config._data))
        self.gemini = genai.Client(
            api_key=self.config["gemini"]["api_key"],
        )

    async def stop(self) -> None:
        pass

    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    @event.on(EventType.ROOM_MESSAGE)  # type: ignore
    async def on_message(self, evt: MessageEvent) -> None:
        room_alias = await self._get_room_alias(evt.room_id)
        if evt.content.msgtype != MessageType.NOTICE:
            # self.log.debug(
            #     f"Ignoring message of type {evt.content.msgtype} from {evt.sender} in {room_alias or evt.room_id}"
            # )
            return
        if evt.sender != self.config["meetbot_id"]:
            self.log.debug(
                f"Ignoring message from {evt.sender} in {room_alias or evt.room_id}, I'm only listening to {self.config['meetbot_id']}"
            )
            return

        await evt.mark_read()

        message_body = cast(str, evt.content.body)
        if not message_body.startswith(MEETBOT_PREFIX):
            self.log.debug(f"Ignoring message from meetbot: {message_body}")
            return

        url = message_body[len(MEETBOT_PREFIX) :]
        room_name = str(room_alias or evt.room_id)
        if room_name != evt.room_id:
            room_name = f"{room_name} ({evt.room_id})"
        self.log.info(
            f"Detected Text Log message from meetbot in {room_name} with URL: {url}"
        )
        await self.handle_meeting_log(evt, url)

    async def handle_meeting_log(self, evt: MessageEvent, url: str) -> None:
        # reaction_event_id = await evt.react("👀")
        await self.client.set_typing(evt.room_id, timeout=30000)
        try:
            meeting_log = await self.get_meeting_log(evt, url)
        except MeetingLogFetchingError as e:
            await evt.reply(str(e))
            return
        # Extract Matrix usernames from the meeting log
        usernames = self.extract_usernames(meeting_log)
        # Ask AI to give a summary of the meeting
        summary = await self.get_summary(meeting_log)
        await self.post_summary(evt, summary, usernames, url)
        # Inform that we're done
        # await self.client.redact(evt.room_id, reaction_event_id, reason="done.")
        await self.client.set_typing(evt.room_id, timeout=0)

    async def _get_room_alias(self, room_id: RoomID) -> RoomAlias | None:
        try:
            content = cast(
                CanonicalAliasStateEventContent,
                await self.client.get_state_event(
                    room_id, EventType.ROOM_CANONICAL_ALIAS
                ),
            )
        except errors.request.MNotFound:
            self.log.warning("No room alias for %s", room_id)
            return None
        return content.canonical_alias

    async def get_meeting_log(self, evt: MessageEvent, url: str) -> str:
        self.log.debug(f"Processing meeting log from URL: {url} in room {evt.room_id}")

        try:
            async with self.http.get(url) as response:
                response.raise_for_status()
                doc_data = await response.text()
        except HTTPError as e:
            self.log.exception(f"Failed to fetch meeting log from {url}: {e}")
            raise MeetingLogFetchingError(f"❌ Failed to fetch meeting log: {e}")
        except Exception as e:
            self.log.exception(f"Unexpected error processing meeting log: {e}")
            raise MeetingLogFetchingError(f"❌ Error processing meeting log: {e}")
        self.log.debug(
            f"Successfully fetched meeting log content ({len(doc_data)} characters)"
        )
        return doc_data

    def extract_usernames(self, meeting_log: str) -> list[str]:
        """Extract Matrix usernames from meeting log lines.

        Looks for lines like:
        "2025-09-25 08:26:00 <@username:server.tld> Message content"

        Returns a set of unique Matrix usernames found.
        """
        pattern = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d <(@[^>]+)> "
        matches = re.findall(pattern, meeting_log)
        usernames = set(matches)
        ignored_ids = [self.client.mxid, self.config["meetbot_id"]] + self.config[
            "ignored_participants"
        ]
        for ignored_id in ignored_ids:
            with suppress(KeyError):
                usernames.remove(ignored_id)
        result = list(sorted(usernames))
        self.log.info(f"Found {len(result)} unique participants: {result}")
        return result

    async def get_summary(self, meeting_log: str) -> str | None:
        def _ask_ai() -> GenerateContentResponse:
            return self.gemini.models.generate_content(
                model=self.config["gemini"]["model"],
                contents=[
                    Part.from_bytes(
                        data=meeting_log.encode("utf-8"),
                        mime_type="text/plain",
                    ),
                    LLM_PROMPT,
                ],
            )

        loop = asyncio.get_running_loop()
        response: GenerateContentResponse = await loop.run_in_executor(None, _ask_ai)
        return response.text

    async def post_summary(
        self, evt: MessageEvent, summary: str | None, usernames: list[str], url: str
    ) -> None:
        if summary is None:
            self.log.warning(
                "Could not generate summary for meeting in room %s",
                evt.room_id,
            )
            return
        self.log.info(
            "Sending summary to %s participants in room %s",
            len(usernames),
            evt.room_id,
        )
        response_event_id = await evt.respond(
            RESPONSE_TEMPLATE.format(
                members=", ".join(usernames),
                summary=summary,
            ),
            markdown=True,
        )
        await self.client.react(evt.room_id, response_event_id, "✅")
        await self.client.react(evt.room_id, response_event_id, "❌")

        summary_path = urlparse(url).path
        summary_path = summary_path[: -len(".log.txt")] + ".summary.md"
        summary_path = summary_path.lstrip("/")
        # Store the summary
        await self._save_summary_to_file(
            summary,
            summary_path,
            validated=False,
        )

    async def _save_summary_to_file(
        self, summary: str, path: str, validated: bool = False
    ) -> None:
        """Save the summary to a file in the meetings directory."""
        if not self.config["meetings_directory"]:
            return  # Disabled

        if not validated:
            summary = (
                "*Note: This summary has not been validated by the attendants.*\n\n"
                + summary
            )

        meetings_dir = Path(self.config["meetings_directory"])
        file_path = meetings_dir / path
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Write the summary to the file
        # (even if it already exists, because we always store the unvalidated summary first)
        file_path.write_text(summary)
        self.log.info(f"Saved summary to {file_path}")
