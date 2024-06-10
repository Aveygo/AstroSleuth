use crate::screens::{Model, Frame, List, Block, Modifier, HighlightSpacing, ListState, Style};

pub fn home_screen(model: &mut Model, f: &mut Frame) {

    let src_possible_file = model.src_pth.clone();
    let dst_possible_file = model.dst_pth.clone();

    let items = [format!("{:?}", src_possible_file), format!("{:?}", dst_possible_file)];

    let list = List::new(items)
            .block(Block::bordered().title("List"))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .add_modifier(Modifier::REVERSED)
            )
            .highlight_symbol(">")
            .highlight_spacing(HighlightSpacing::Always);

    let mut state = ListState::default();
    state.select(Some(model.cursor));

    f.render_stateful_widget(list, f.size(), &mut state);
    /* 

    f.render_widget(
        Paragraph::new(LOGO),
        f.size(),
    );

    match src_possible_file {
        Some(src_possible_file) => {
            f.render_widget(
                Paragraph::new(format!("Src selected Path: {:?}", src_possible_file.path())),
                f.size(),
            );
        },
        _ => {
            f.render_widget(
                Paragraph::new(format!("No Src path selected")),
                f.size(),
            );
        }
    }

    match dst_possible_file {
        Some(dst_possible_file) => {
            f.render_widget(
                Paragraph::new(format!("Dst selected Path: {:?}", dst_possible_file.path())),
                f.size(),
            );
        },
        _ => {
            f.render_widget(
                Paragraph::new(format!("No Dst path selected")),
                f.size(),
            );
        }
    }
    */
}