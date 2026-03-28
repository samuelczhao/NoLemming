"""Social action definitions matching OASIS action types."""

from __future__ import annotations

from enum import Enum


class SocialAction(Enum):
    """Available actions an agent can take on the platform."""

    CREATE_POST = "create_post"
    LIKE_POST = "like_post"
    REPOST = "repost"
    QUOTE_POST = "quote_post"
    FOLLOW = "follow"
    DO_NOTHING = "do_nothing"
    CREATE_COMMENT = "create_comment"
