use std::fs;

fn main() {
    let dir = tempfile::tempdir().unwrap();
    let md_path = dir.path().join("test.md");
    fs::write(&md_path, "test").unwrap();
    
    println!("Tempdir: {:?}", dir.path());
    let mut walker = walkdir::WalkDir::new(dir.path()).into_iter();
    while let Some(Ok(entry)) = walker.next() {
        println!("Entry: {:?}", entry.path());
        println!("Hidden: {}", entry.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false));
    }
}
