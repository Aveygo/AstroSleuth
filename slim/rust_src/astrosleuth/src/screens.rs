use std::fs::File;

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{prelude::*, widgets::*};

mod explorer;
mod home;

const LOGO: &str = "
    ___         __            _____ __           __  __  
   /   |  _____/ /__________ / ___// /__  __  __/ /_/ /_ 
  / /| | / ___/ __/ ___/ __ \\\\__ \\/ / _ \\/ / / / __/ __ \\
 / ___ |(__  ) /_/ /  / /_/ /__/ / /  __/ /_/ / /_/ / / /
/_/  |_/____/\\__/_/   \\____/____/_/\\___/\\__,_/\\__/_/ /_/ 
";

#[derive(Debug, Default)]
struct Model {
    cursor: usize,
    src_pth: Option<std::fs::File>,
    dst_pth: Option<std::fs::File>,
    running_state: Screen,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum Screen {
    #[default]
    Running,
    FileExplorer,
    Done,
}

#[derive(PartialEq)]
enum Message {
    Quit,
    ESC,
    ENTER,

    DOWN,
    UP,
    TAB
}


pub fn tui() -> color_eyre::Result<()> {
    tui::install_panic_hook();
    let mut terminal = tui::init_terminal()?;
    let mut model = Model::default();

    
    while model.running_state != Screen::Done {
        match model.running_state {
            Screen::Running => {
                terminal.draw(|f| home::home_screen(&mut model, f))?;
            },
            Screen::FileExplorer => {
                terminal.draw(|f| explorer::explorer_screen(&mut model, f))?;
            },
            Screen::Done => {}
        }
        
        let mut current_msg = handle_event(&mut model)?;

        while current_msg.is_some() {
            current_msg = update(&mut model, current_msg.unwrap());
        }
    }

    tui::restore_terminal()?;
    Ok(())
}

fn handle_event(model: &mut Model) -> color_eyre::Result<Option<Message>> {
    let event = event::read()?;

    match model.running_state {
        Screen::FileExplorer => {
            if let Some(ref mut explorer) = model.file_explorer {
                explorer.handle(&event).unwrap();
            } else {
                panic!("File explorer is not initialized.");
            }
        }
        _ => {}
    }

    if let Event::Key(key) = event {
        if key.kind == event::KeyEventKind::Press {
            return Ok(handle_key(key));
        }
    }

    Ok(None)
}

fn handle_key(key: event::KeyEvent) -> Option<Message> {
    match key.code {
        KeyCode::Esc => Some(Message::ESC),
        KeyCode::Enter => Some(Message::ENTER),
        
        KeyCode::Down => Some(Message::DOWN),
        KeyCode::Up => Some(Message::UP),
        KeyCode::Tab => Some(Message::TAB),

        KeyCode::Char('c') => {
            match key.modifiers {
                KeyModifiers::CONTROL => {
                    Some(Message::Quit)
                },
                _ => {None},
            }
        },
        _ => None,
    }
}

fn update(model: &mut Model , msg: Message) -> Option<Message> {

    match msg {
        Message::Quit => {
            model.running_state = Screen::Done;
        },
        _ => {}
    }

    match model.running_state {
        
        Screen::FileExplorer => {
            match msg {
                Message::ENTER => {
                    model.running_state = Screen::Running;

                    if let Some(ref mut explorer) = model.file_explorer {
                        let selected_file_id = explorer.selected_idx();
                        let selected_file = explorer.files().get(selected_file_id);

                        match selected_file {
                            Some(selected_file) => {
                                if model.cursor == 0 {
                                    model.src_pth = Some(selected_file.clone());
                                } else if model.cursor == 1 {
                                    model.dst_pth = Some(selected_file.clone());
                                }
                            },
                            _ => {}
                        }
                    } else {
                        panic!("File explorer is not initialized.");
                    }
                },
                Message::ESC => {
                    model.running_state = Screen::Running;
                },
                _ => {}
            };
            
        },

        Screen::Running => {
            match msg {

                Message::UP => {
                    model.cursor = model.cursor - 1;
                    model.cursor = model.cursor.max(0).min(2);
                },
                Message::DOWN => {
                    model.cursor = model.cursor + 1;
                    model.cursor = model.cursor.max(0).min(2);
                },
                Message::ENTER => {
                    if model.cursor < 2 {
                        let file_explorer = FileExplorer::with_theme(Theme::default().add_default_title()).unwrap();
                        model.file_explorer = Some(file_explorer);
                        model.running_state = Screen::FileExplorer;
                    }
                }

                _ => {}
            };
        }
        _ => {}
    }

    None
}

mod tui {
    use crossterm::{
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        ExecutableCommand,
    };
    use ratatui::prelude::*;
    use std::{io::stdout, panic};

    pub fn init_terminal() -> color_eyre::Result<Terminal<impl Backend>> {
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
        Ok(terminal)
    }

    pub fn restore_terminal() -> color_eyre::Result<()> {
        stdout().execute(LeaveAlternateScreen)?;
        disable_raw_mode()?;
        Ok(())
    }

    pub fn install_panic_hook() {
        let original_hook = panic::take_hook();
        panic::set_hook(Box::new(move |panic_info| {
            stdout().execute(LeaveAlternateScreen).unwrap();
            disable_raw_mode().unwrap();
            original_hook(panic_info);
        }));
    }
}