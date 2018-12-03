extern crate regex;

use regex::Regex;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io;
use std::io::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 {
        let day = args[1].parse::<u32>().unwrap();

        match run_day(day) {
            Ok(res) => println!("Day {}: {}", day, res),
            Err(_) => println!("Error, can't run day {}", day),
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
    let mut contents = String::new();
    File::open(format!("input{}.txt", i))?.read_to_string(&mut contents)?;
    Ok(contents)
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
    x.chars()
        .any(|c| x.chars().filter(|&a| a == c).count() == i)
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

struct Box {
    id: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

pub fn day3(input: &str) -> (usize, u32) {
    let re = Regex::new(r"(?m)^#(\d+) @ (\d+),(\d+): (\d+)x(\d+)").unwrap();
    let boxes: Vec<Box> = re
        .captures_iter(input)
        .map(|m| Box {
            id: m[1].parse().unwrap(),
            x: m[2].parse().unwrap(),
            y: m[3].parse().unwrap(),
            w: m[4].parse().unwrap(),
            h: m[5].parse().unwrap(),
        }).collect();

    let mut claimed: HashSet<(u32, u32)> = HashSet::new();
    let mut twice: HashSet<(u32, u32)> = HashSet::new();

    for b in &boxes {
        for i in b.x..b.x + b.w {
            for j in b.y..b.y + b.h {
                if !claimed.insert((i, j)) {
                    twice.insert((i, j));
                }
            }
        }
    }

    let id: u32 = boxes
        .iter()
        .find(|b| !(b.x..b.x + b.w).any(|x| (b.y..b.y + b.h).any(|y| twice.contains(&(x, y)))))
        .map(|b| b.id)
        .unwrap();

    (twice.len(), id)
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

    #[test]
    fn test_day3() {
        let input = read_input(3).unwrap();
        let (a, b) = day3(&input);
        assert_eq!(a, 101565);
        assert_eq!(b, 656);
    }
}
