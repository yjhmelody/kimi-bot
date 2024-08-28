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
    let openai_model = env::var("OPENAI_MODEL").unwrap_or("moonshot-v1-8k".to_string());

    let config = Config {
        openai_key,
        openai_model,
        openai_url,
    };

    let openai_config = config.to_openai_config();
    let client = Client::with_config(openai_config);

    let state = State::new_shared(State::new(client).with_config(config));
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
    config: Config,
}

#[derive(Default, Debug, Clone)]
pub struct Config {
    openai_key: String,
    openai_url: String,
    openai_model: String,
}

impl Config {
    pub fn to_openai_config(&self) -> OpenAIConfig {
        let config = OpenAIConfig::new()
            .with_api_key(self.openai_key.clone())
            .with_api_base(self.openai_url.clone())
            .with_org_id("the-continental");

        config
    }
}

impl State {
    pub fn new_shared(state: Self) -> SharedState {
        Arc::new(RwLock::new(state))
    }

    pub fn new(client: Client<OpenAIConfig>) -> Self {
        Self {
            client,
            ..Self::default()
        }
    }

    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }
}

#[derive(BotCommands, Clone)]
#[command(rename_rule = "kebab-case", description = "These commands are supported:")]
enum Command {
    #[command(description = "display this text.")]
    Help,
    #[command(description = "Show current config")]
    CurrentConfig,
    #[command(description = "chat with kimi.")]
    Kimi(String),
    #[command(description = "Update the endpoint config of kimi")]
    UpdateEndpoint(String),
    #[command(description = "Update the model config of kimi")]
    UpdateModel(String),
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
            Command::CurrentConfig => {
                let response = {
                    let state = state.read().await;
                    let config = &state.config;
                    format!("config: url({}), model({})", config.openai_url, config.openai_model)
                };
                bot.send_message(id, response).await?
            },
            Command::Kimi(text) => {
                let model = {
                    let state = state.read().await;
                    state.config.openai_model.clone()
                };
                let request = CreateChatCompletionRequestArgs::default()
                    .model(model)
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
            Command::UpdateEndpoint(endpiont) => {
                {
                    let mut state = state.write().await;
                    state.config.openai_url = endpiont;
                    // create a new config
                    let openai_config = state.config.to_openai_config();
                    state.client = Client::with_config(openai_config);
                    log::debug!("The config: {:#?}", state.config);
                }
                bot.send_message(id, "Update endpoint successful.").await?
            }
            Command::UpdateModel(model) => {
                {
                    let mut state = state.write().await;
                    state.config.openai_model = model;
                    // create a new config
                    let openai_config = state.config.to_openai_config();
                    state.client = Client::with_config(openai_config);
                    log::debug!("The config: {:#?}", state.config);
                }
                bot.send_message(id, "Update model successful.").await?
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
