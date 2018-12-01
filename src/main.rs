use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 {
        let day = args[1].parse::<u32>().unwrap();

        if let Some(res) = run_day(day) {
            println!("Day {}: {}", day, res);
        } else {
            println!("Error, can't run day {}", day);
        }
    } else {
        for i in 1..=24 {
            if let Some(res) = run_day(i) {
                println!("Day {}: {}", i, res);
            }
        }
    }
}

fn run_day(i: u32) -> Option<String> {
    let mut days: Vec<Box<Fn(&str) -> String>> = vec![];
    days.push(Box::new(|input| format!("{:?}", day1(input))));
    days.push(Box::new(|input| format!("{:?}", day2(input))));
    days.push(Box::new(|input| format!("{:?}", day3(input))));
    days.push(Box::new(|input| format!("{:?}", day4(input))));
    days.push(Box::new(|input| format!("{:?}", day5(input))));

    if let Some(input) = read_input(i) {
        Some(days[i as usize - 1](&input))
    } else {
        None
    }
}

fn read_input(i: u32) -> Option<String> {
    match File::open(format!("input{}.txt", i)) {
        Ok(mut f) => {
            let mut contents = String::new();
            f.read_to_string(&mut contents).unwrap();
            Some(contents)
        }
        _ => None,
    }
}

pub fn day1(input: &str) -> (i32, i32) {
    let changes: Vec<i32> = input.lines().map(|l| l.parse::<i32>().unwrap()).collect();
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

pub fn day2(_input: &str) -> (i32, i32) {
    (-1, -1)
}

pub fn day3(_input: &str) -> (i32, i32) {
    (-1, -1)
}

pub fn day4(_input: &str) -> (i32, i32) {
    (-1, -1)
}

pub fn day5(_input: &str) -> (i32, i32) {
    (-1, -1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_day1() {
        let input = read_input(1).unwrap();
        let (a, b) = day1(&input);
        assert_eq!(a, 502);
        assert_eq!(b, 71961);
    }

}
