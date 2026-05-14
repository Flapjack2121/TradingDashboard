"""
alerts - Multi-channel notification dispatcher.

Module:
    notifier   `AlertDispatcher` fans out events to the channels enabled
               in the .env file. Each channel is a `Notifier` subclass:

                   ConsoleNotifier   stdout / logger (always available)
                   TelegramNotifier  Bot API POST (TELEGRAM_BOT_TOKEN, CHAT_ID)
                   DiscordNotifier   webhook POST (DISCORD_WEBHOOK_URL)

               Network failures are caught and logged so alert outages
               never block the trading pipeline.

Import directly:

    from alerts.notifier import (
        AlertDispatcher,
        AlertEvent,
        ConsoleNotifier,
        TelegramNotifier,
        DiscordNotifier,
    )
"""
