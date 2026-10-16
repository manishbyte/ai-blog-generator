"""BlogForge Streamlit user interface."""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Any

import streamlit as st

from ui.api_client import APIError, BlogForgeAPI


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="BlogForge",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ChatGPT-inspired styling
# ============================================================

st.markdown(
    """
    <style>
    /* Keep the conversation readable and centered, like a chat workspace. */
    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 7.5rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        line-height: 1.7;
    }

    /* Streamlit's chat input is pinned to the bottom of the viewport. */
    [data-testid="stChatInput"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 24px !important;
        padding: 0.85rem 1rem !important;
    }

    [data-testid="stChatInput"] button {
        border-radius: 50% !important;
    }

    [data-testid="stForm"] button,
    [data-testid="stButton"] button {
        border-radius: 10px;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-bottom: 6.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

DEFAULT_STATE: dict[str, Any] = {
    "access_token": None,
    "user": None,
    "conversation_id": None,
    "conversations": [],
    "pending_stream": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_auth_state() -> None:
    """Clear authentication and conversation-related UI state."""

    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def get_api() -> BlogForgeAPI:
    """Build an API client using the current access token."""

    return BlogForgeAPI(
        token=st.session_state.get("access_token")
    )


def show_api_error(exc: Exception) -> None:
    """Show a user-friendly API error."""

    st.error(str(exc))


# ============================================================
# Blog Markdown and generated image rendering
# ============================================================

# Matches the Markdown produced by the image agent:
# ![Image description](images/example.png)
GENERATED_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(\s*images/([^)]+?)\s*\)"
)


def render_blog_markdown(
    content: str,
    api: BlogForgeAPI,
) -> None:
    """
    Render blog Markdown and show generated images inline.

    Generated image files are fetched through FastAPI so the user's
    bearer token is sent with each image request.
    """

    if not content:
        st.info("This message has no content.")
        return

    position = 0

    for match in GENERATED_IMAGE_PATTERN.finditer(content):
        # Render the Markdown before this image.
        text_before = content[position:match.start()]

        if text_before.strip():
            st.markdown(
                text_before,
                unsafe_allow_html=False,
            )

        alt_text = match.group(1).strip()
        filename = match.group(2).strip().strip("\"'")

        try:
            image_bytes = api.get_generated_image(filename)

            st.image(
                image_bytes,
                caption=alt_text or None,
                use_container_width=True,
            )

        except APIError as exc:
            st.warning(
                f"Could not load generated image "
                f"`{filename}`: {exc}"
            )

        position = match.end()

    # Render everything after the last generated image.
    remaining = content[position:]

    if remaining.strip():
        st.markdown(
            remaining,
            unsafe_allow_html=False,
        )



def stream_blog_markdown(
    content: str,
    api: BlogForgeAPI,
    chunk_size: int = 35,
    delay_seconds: float = 0.008,
) -> None:
    """Reveal completed blog text progressively, then render images inline.

    This is UI-only streaming: the backend has already completed generation
    before this function runs.
    """

    if not content:
        st.info("This message has no content.")
        return

    placeholder = st.empty()

    # Reveal the Markdown progressively. The final render below replaces this
    # temporary plain Markdown view so generated images load inline.
    for end in range(chunk_size, len(content), chunk_size):
        placeholder.markdown(
            content[:end],
            unsafe_allow_html=False,
        )
        time.sleep(delay_seconds)

    placeholder.markdown(content, unsafe_allow_html=False)
    time.sleep(delay_seconds)
    placeholder.empty()

    render_blog_markdown(content, api)


# ============================================================
# Authentication screen
# ============================================================

def render_auth_screen() -> None:
    """Render registration and login forms."""

    st.title("✍️ BlogForge")
    st.caption(
        "Research, write, and illustrate your next blog with AI."
    )

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        login_tab, register_tab = st.tabs(
            ["Log in", "Create account"]
        )

        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input(
                    "Email",
                    key="login_email",
                )
                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )

                login_submitted = st.form_submit_button(
                    "Log in",
                    type="primary",
                    use_container_width=True,
                )

            if login_submitted:
                if not login_email.strip() or not login_password:
                    st.warning(
                        "Enter your email and password."
                    )
                else:
                    try:
                        unauthenticated_api = BlogForgeAPI()
                        token_data = unauthenticated_api.login(
                            login_email.strip(),
                            login_password,
                        )

                        token = token_data["access_token"]
                        authenticated_api = BlogForgeAPI(token=token)
                        current_user = authenticated_api.me()

                        st.session_state["access_token"] = token
                        st.session_state["user"] = current_user
                        st.session_state["conversation_id"] = None
                        st.session_state["conversations"] = []

                        st.rerun()

                    except (APIError, KeyError) as exc:
                        st.error(f"Login failed: {exc}")

        with register_tab:
            with st.form("register_form"):
                register_email = st.text_input(
                    "Email",
                    key="register_email",
                )
                register_password = st.text_input(
                    "Password",
                    type="password",
                    key="register_password",
                    help="Use at least 8 characters.",
                )
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="confirm_password",
                )

                register_submitted = st.form_submit_button(
                    "Create account",
                    type="primary",
                    use_container_width=True,
                )

            if register_submitted:
                if not register_email.strip():
                    st.warning("Enter your email address.")
                elif len(register_password) < 8:
                    st.warning(
                        "Your password must contain at least "
                        "8 characters."
                    )
                elif register_password != confirm_password:
                    st.warning("The passwords do not match.")
                else:
                    try:
                        BlogForgeAPI().register(
                            register_email.strip(),
                            register_password,
                        )

                        st.success(
                            "Account created successfully. "
                            "You can now log in."
                        )

                    except APIError as exc:
                        st.error(
                            f"Registration failed: {exc}"
                        )


# ============================================================
# Authentication validation
# ============================================================

def validate_session() -> bool:
    """Validate the saved token against FastAPI."""

    token = st.session_state.get("access_token")

    if not token:
        return False

    try:
        user = get_api().me()
        st.session_state["user"] = user
        return True

    except APIError:
        reset_auth_state()
        return False


# ============================================================
# Sidebar
# ============================================================

def load_conversations(
    api: BlogForgeAPI,
) -> list[dict[str, Any]]:
    """Fetch conversations and update session state."""

    conversations = api.list_conversations()
    st.session_state["conversations"] = conversations
    return conversations


def render_sidebar(api: BlogForgeAPI) -> None:
    """Render conversation navigation and account actions."""

    with st.sidebar:
        st.title("✍️ BlogForge")
        st.caption("Your AI writing workspace")

        user = st.session_state.get("user") or {}
        if user.get("email"):
            st.caption(f"Signed in as {user['email']}")

        st.divider()

        if st.button(
            "＋ New blog",
            use_container_width=True,
            type="primary",
        ):
            try:
                conversation = api.create_conversation()

                st.session_state["conversation_id"] = (
                    conversation["id"]
                )

                # Refresh the sidebar after creating the conversation.
                load_conversations(api)
                st.rerun()

            except APIError as exc:
                show_api_error(exc)

        st.subheader("Your conversations")

        try:
            conversations = load_conversations(api)
        except APIError as exc:
            show_api_error(exc)
            conversations = st.session_state.get(
                "conversations", []
            )

        if not conversations:
            st.caption("Your conversations will appear here.")

        for conversation in conversations:
            conversation_id = conversation["id"]
            title = conversation.get("title") or "New blog"

            is_selected = (
                st.session_state.get("conversation_id")
                == conversation_id
            )

            button_label = (
                f"● {title}" if is_selected else title
            )

            if st.button(
                button_label,
                key=f"conversation_{conversation_id}",
                use_container_width=True,
                help=title,
            ):
                st.session_state["conversation_id"] = (
                    conversation_id
                )
                st.rerun()

        selected_id = st.session_state.get("conversation_id")

        if selected_id:
            st.divider()

            if st.button(
                "Delete selected conversation",
                use_container_width=True,
            ):
                try:
                    api.delete_conversation(selected_id)

                    st.session_state["conversation_id"] = None
                    load_conversations(api)
                    st.rerun()

                except APIError as exc:
                    show_api_error(exc)

        st.divider()

        if st.button(
            "Log out",
            use_container_width=True,
        ):
            try:
                # Revoke the JWT on the backend before clearing the UI.
                api.logout()
            except APIError as exc:
                st.error(
                    "The server could not confirm token revocation. "
                    f"{exc}"
                )
            finally:
                # Clear the token from this Streamlit session either way.
                reset_auth_state()
                st.rerun()


# ============================================================
# Conversation display
# ============================================================

def render_message_attachments(
    message: dict[str, Any],
    api: BlogForgeAPI,
) -> None:
    """Render any database attachments associated with a message."""

    attachments = message.get("attachments") or []

    for attachment in attachments:
        attachment_id = attachment.get("id")
        mime_type = attachment.get("mime_type", "")

        if not attachment_id:
            continue

        try:
            if mime_type.startswith("image/"):
                image_bytes = api.get_attachment(attachment_id)

                st.image(
                    image_bytes,
                    caption=attachment.get("alt_text")
                    or attachment.get("filename"),
                    use_container_width=True,
                )
            else:
                st.caption(
                    f"Attachment: "
                    f"{attachment.get('filename', 'file')}"
                )

        except APIError as exc:
            st.warning(
                f"Could not load attachment: {exc}"
            )


def render_conversation(
    conversation_id: str,
    api: BlogForgeAPI,
) -> None:
    """Load and display a conversation's messages."""

    try:
        conversation = api.get_conversation(conversation_id)

    except APIError as exc:
        show_api_error(exc)
        return

    st.title(conversation.get("title") or "Blog workspace")
    st.caption(
        "Generate a blog, review the content, and view its images inline."
    )

    messages = conversation.get("messages") or []
    pending_stream = st.session_state.get("pending_stream")
    streamed_pending_message = False

    # If generation just completed, progressively reveal that saved assistant
    # message in its normal position in the conversation history.
    pending_matches_conversation = (
        isinstance(pending_stream, dict)
        and pending_stream.get("conversation_id") == conversation_id
    )
    pending_message_id = (
        pending_stream.get("message_id")
        if pending_matches_conversation
        else None
    )
    pending_content = (
        pending_stream.get("content")
        if pending_matches_conversation
        else None
    )

    for message in messages:
        role = message.get("role", "assistant")

        if role not in {"user", "assistant"}:
            role = "assistant"

        content = message.get("content", "")
        message_id = message.get("id")
        should_stream = (
            pending_matches_conversation
            and not streamed_pending_message
            and role == "assistant"
            and (
                (pending_message_id is not None and message_id == pending_message_id)
                or (pending_message_id is None and content == pending_content)
            )
        )

        with st.chat_message(role):
            if should_stream:
                stream_blog_markdown(content, api)
                streamed_pending_message = True
            elif role == "assistant":
                render_blog_markdown(content, api)
            else:
                st.markdown(
                    content,
                    unsafe_allow_html=False,
                )

            render_message_attachments(message, api)

    if pending_matches_conversation:
        # Clear after one display so later reruns show the saved message
        # normally rather than replaying the animation.
        st.session_state["pending_stream"] = None

    # Keep optional research-date controls just above the sticky composer.
    with st.expander("Research settings", expanded=False):
        as_of_date = st.date_input(
            "Research reference date",
            value=date.today(),
            help=(
                "The workflow can use this date when planning "
                "time-sensitive research."
            ),
            key=f"as_of_date_{conversation_id}",
        )

    # st.chat_input is Streamlit's native bottom-pinned chat composer.
    topic = st.chat_input(
        "Message BlogForge… What should your blog be about?",
        max_chars=10_000,
        key=f"chat_input_{conversation_id}",
    )

    if topic is not None:
        if not topic.strip():
            st.warning("Enter a topic before generating a blog.")
            return

        try:
            with st.spinner(
                "BlogForge is researching, writing, and generating images. "
                "This may take several minutes."
            ):
                api.generate_blog(
                    conversation_id=conversation_id,
                    topic=topic.strip(),
                    as_of=as_of_date.isoformat(),
                )

            # Fetch the persisted result. UI-only streaming starts after the
            # backend has finished; no API or backend changes are required.
            updated_conversation = api.get_conversation(conversation_id)
            updated_messages = updated_conversation.get("messages") or []
            latest_assistant = next(
                (
                    message
                    for message in reversed(updated_messages)
                    if message.get("role") == "assistant"
                ),
                None,
            )

            if latest_assistant is None:
                st.warning(
                    "Generation finished, but no assistant message was "
                    "found in the saved conversation."
                )
                st.rerun()

            st.session_state["pending_stream"] = {
                "conversation_id": conversation_id,
                "message_id": latest_assistant.get("id"),
                "content": latest_assistant.get("content", ""),
            }
            st.rerun()

        except APIError as exc:
            show_api_error(exc)


# ============================================================
# Main application
# ============================================================

def main() -> None:
    """Run the BlogForge Streamlit application."""

    if not validate_session():
        render_auth_screen()
        return

    api = get_api()

    render_sidebar(api)

    conversation_id = st.session_state.get("conversation_id")

    if conversation_id:
        render_conversation(conversation_id, api)
    else:
        st.title("Welcome to BlogForge")
        st.write(
            "Create a new conversation to start writing a blog."
        )

        st.markdown(
            """
            **Your workflow**

            - Enter a topic for your blog.
            - Let the existing LangGraph workflow research and write it.
            - View generated images inline with the blog.
            - Reopen saved conversations from the sidebar.
            """
        )

        if st.button(
            "＋ Start a new blog",
            type="primary",
        ):
            try:
                conversation = api.create_conversation()

                st.session_state["conversation_id"] = (
                    conversation["id"]
                )

                load_conversations(api)
                st.rerun()

            except APIError as exc:
                show_api_error(exc)


if __name__ == "__main__":
    main()