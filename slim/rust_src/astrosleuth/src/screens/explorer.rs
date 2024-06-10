use crate::screens::{Model, Frame};

pub fn explorer_screen(model: &mut Model, f: &mut Frame) {
    let explorer = &model.file_explorer;
    match explorer {
        Some(explorer) => {
            f.render_widget(&explorer.widget(), f.size())
        },
        _ => panic!("")
    }
}