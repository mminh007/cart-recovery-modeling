"""Vietnamese prompt templates per strategy and outreach channel for LLM SFT/DPO."""

from __future__ import annotations

CHANNEL_CHAR_LIMITS: dict[str, int] = {
    "push": 100,
    "sms": 160,
    "zalo": 400,
    "email": 1200,
}

SYSTEM_PROMPT_VI = (
    "Bạn là trợ lý chăm sóc khách hàng thông minh của sàn thương mại điện tử tại Việt Nam. "
    "Nhiệm vụ của bạn là viết tin nhắn nhắc giỏ hàng tự nhiên, thân thiện, đúng tuyệt đối tên sản phẩm, "
    "giá tiền và mức ưu đãi được cung cấp. Tuyệt đối không bịa thêm mã giảm giá hay freeship không có trong dữ liệu."
)


def build_recovery_user_prompt(
    customer_name: str,
    segment: str,
    strategy: str,
    channel: str,
    items_summary: str,
    cart_total_vnd: int,
    allowed_discount_pct: int,
) -> str:
    """Format a structured prompt for Vietnamese cart recovery message generation."""
    max_chars = CHANNEL_CHAR_LIMITS.get(channel.lower(), 400)
    return (
        f"Tạo tin nhắn khôi phục giỏ hàng cho khách hàng:\n"
        f"- Tên khách: {customer_name}\n"
        f"- Phân khúc: {segment}\n"
        f"- Chiến lược tiếp cận: {strategy}\n"
        f"- Kênh gửi: {channel} (giới hạn tối đa {max_chars} ký tự)\n"
        f"- Sản phẩm trong giỏ: {items_summary}\n"
        f"- Tổng giá trị giỏ: {cart_total_vnd:,}đ\n"
        f"- Ưu đãi tối đa được phép: {allowed_discount_pct}%\n"
    )
