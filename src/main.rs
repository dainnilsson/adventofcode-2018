use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io;
use std::io::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 {
        let day = args[1].parse::<u32>().unwrap();

        if let Ok(res) = run_day(day) {
            println!("Day {}: {}", day, res);
        } else {
            println!("Error, can't run day {}", day);
        }
    } else {
        for i in 1..=24 {
            if let Ok(res) = run_day(i) {
                println!("Day {}: {}", i, res);
            }
        }
    }
}

fn run_day(i: u32) -> io::Result<String> {
    read_input(i).map(|input| match i {
        1 => format!("{:?}", day1(&input)),
        2 => format!("{:?}", day2(&input)),
        3 => format!("{:?}", day3(&input)),
        4 => format!("{:?}", day4(&input)),
        5 => format!("{:?}", day5(&input)),
        _ => panic!("Day not implemented!"),
    })
}

fn read_input(i: u32) -> io::Result<String> {
    File::open(format!("input{}.txt", i)).map(|mut f| {
        let mut contents = String::new();
        f.read_to_string(&mut contents).unwrap();
        contents
    })
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

fn duplicates(x: &str, i: usize) -> bool {
    x.chars().any(|c| x.chars().filter(|&a| a == c).count() == i)
}

pub fn day2(input: &str) -> (usize, String) {
    let two = input.lines().filter(|l| duplicates(l, 2)).count();
    let three = input.lines().filter(|l| duplicates(l, 3)).count();

    let (a, b) = input
        .lines()
        .find_map(|a| {
            input
                .lines()
                .find(|b| a.chars().zip(b.chars()).filter(|(x, y)| x != y).count() == 1)
                .map(|b| (a, b))
        }).unwrap();

    let common: String = a
        .chars()
        .zip(b.chars())
        .filter(|(x, y)| x == y)
        .map(|(x, _)| x)
        .collect();

    (two * three, common)
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

    #[test]
    fn test_day2() {
        let input = read_input(2).unwrap();
        let (a, b) = day2(&input);
        assert_eq!(a, 5976);
        assert_eq!(b, "xretqmmonskvzupalfiwhcfdb");
    }
}
