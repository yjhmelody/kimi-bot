use std::env;
use std::sync::Arc;

use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, CreateChatCompletionRequestArgs,
};
use async_openai::Client;
use teloxide::{filter_command, prelude::*, update_listeners, utils::command::BotCommands};
use teloxide::{ApiError, RequestError};
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    pretty_env_logger::init();
    log::info!("Starting command bot...");

    let openai_key = env::var("OPENAI_API_KEY").expect("Missing OPENAI_API_KEY env variable");
    let openai_url = env::var("OPENAI_API_URL").expect("Missing OPENAI_API_URL env variable");

    let config = OpenAIConfig::new()
        .with_api_key(openai_key)
        .with_api_base(openai_url)
        .with_org_id("the-continental");
    let client = Client::with_config(config);

    let state = State::new_shared(client);
    let bot = Bot::from_env();

    let ignore_update = |_upd| Box::pin(async {});
    let listener = update_listeners::polling_default(bot.clone()).await;
    Dispatcher::builder(
        bot,
        Update::filter_message()
            .chain(filter_command::<Command, _>())
            .endpoint(answer),
    )
    .default_handler(ignore_update)
    .dependencies(dptree::deps![state])
    .enable_ctrlc_handler()
    .build()
    .dispatch_with_listener(
        listener,
        LoggingErrorHandler::with_custom_text("An error from the update listener"),
    )
    .await;
}

pub type SharedState = Arc<RwLock<State>>;

#[derive(Default)]
pub struct State {
    messages_total: u64,
    client: Client<OpenAIConfig>,
}

impl State {
    pub fn new_shared(client: Client<OpenAIConfig>) -> SharedState {
        Arc::new(RwLock::new(State::new(client)))
    }

    pub fn new(client: Client<OpenAIConfig>) -> Self {
        Self {
            messages_total: 0,
            client,
        }
    }
}

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "These commands are supported:")]
enum Command {
    #[command(description = "display this text.")]
    Help,
    #[command(description = "chat with kimi.")]
    Kimi(String),
}

async fn answer(bot: Bot, state: SharedState, cmd: Command, msg: Message) -> ResponseResult<()> {
    {
        let mut state = state.write().await;
        state.messages_total += 1;
    }
    let id = msg.chat.id;
    let f = async {
        match cmd {
            Command::Help => bot.send_message(id, Command::descriptions().to_string()).await?,
            Command::Kimi(text) => {
                let request = CreateChatCompletionRequestArgs::default()
                    // must use moonshot model
                    .model("moonshot-v1-8k")
                    .messages([ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                        content: text.into(),
                        ..Default::default()
                    })])
                    .build()
                    .map_err(unknown_error)?;

                let response = {
                    let state = state.read().await;
                    state.client.chat().create(request).await.map_err(unknown_error)?
                };

                let reply = response.choices[0]
                    .message
                    .content
                    .clone()
                    .ok_or(unknown_error("The remote chat AI response failed"))?;
                bot.send_message(id, reply).await?
            }
        };

        Ok::<(), RequestError>(())
    };

    if let Err(err) = f.await {
        bot.send_message(id, format!("Error occurred: {err}")).await?;
    }

    Ok(())
}

fn unknown_error<T: ToString>(err: T) -> RequestError {
    RequestError::Api(ApiError::Unknown(err.to_string()))
}
