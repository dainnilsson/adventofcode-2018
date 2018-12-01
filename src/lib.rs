use std::fs::File;
use std::io::prelude::*;

pub fn read_input(i: u32) -> String {
    let mut f = File::open(format!("input{}.txt", i)).unwrap();
    let mut contents = String::new();
    f.read_to_string(&mut contents).unwrap();

    contents
}

pub mod day1 {
    use std::collections::HashSet;

    pub fn run(lines: &Vec<&str>) -> (i32, i32) {
        let changes: Vec<i32> = lines.iter().map(|l| l.parse::<i32>().unwrap()).collect();
        let mut seen = HashSet::new();
        let mut f = 0;
        loop {
            for c in &changes {
                f += c;
                if !seen.insert(f) {
                    return (changes.iter().sum(), f);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day1() {
        let input = read_input(1);
        let (a, b) = day1::run(&input.lines().collect());
        assert_eq!(a, 502);
        assert_eq!(b, 71961);
    }

}
