extern crate regex;

use regex::Regex;
use std::collections::{HashMap, HashSet};
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

impl Box {
    fn points<'a>(&'a self) -> impl Iterator<Item = (u32, u32)> + 'a {
        (self.x..self.x + self.w).flat_map(move |x| (self.y..self.y + self.h).map(move |y| (x, y)))
    }
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
        for (x, y) in b.points() {
            if !claimed.insert((x, y)) {
                twice.insert((x, y));
            }
        }
    }

    let id: u32 = boxes
        .iter()
        .find(|b| !b.points().any(|p| twice.contains(&p)))
        .map(|b| b.id)
        .unwrap();

    (twice.len(), id)
}

#[derive(Ord, PartialOrd, PartialEq, Eq)]
enum Day4Action {
    Begin(u32),
    Sleep,
    Wake,
}

#[derive(Ord, PartialOrd, PartialEq, Eq)]
struct Day4Entry {
    year: u32,
    month: u32,
    day: u32,
    h: u32,
    m: u32,
    action: Day4Action,
}

fn max_minute(input: &Vec<(u32, u32)>) -> (u32, u32) {
    let mut ms = HashMap::new();
    input
        .iter()
        .flat_map(|(f, t)| *f..*t)
        .for_each(|x| *ms.entry(x).or_insert(0) += 1);
    ms.iter().max().map(|(&f, &m)| (f, m)).unwrap()
}

pub fn day4(input: &str) -> (u32, u32) {
    let re_entry = Regex::new(r"(?m)^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})\] (.+)$").unwrap();
    let re_id = Regex::new(r".*#(\d+) .*").unwrap();
    let mut entries: Vec<Day4Entry> = re_entry
        .captures_iter(input)
        .map(|m| Day4Entry {
            year: m[1].parse().unwrap(),
            month: m[2].parse().unwrap(),
            day: m[3].parse().unwrap(),
            h: m[4].parse().unwrap(),
            m: m[5].parse().unwrap(),
            action: match &m[6] {
                "wakes up" => Day4Action::Wake,
                "falls asleep" => Day4Action::Sleep,
                _ => Day4Action::Begin(re_id.captures(&m[6]).unwrap()[1].parse().unwrap()),
            },
        }).collect();

    entries.sort();

    let mut id = 0;
    let mut sleep_at = 0;
    let mut sleeps: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for e in &entries {
        match e.action {
            Day4Action::Begin(x) => id = x,
            Day4Action::Sleep => sleep_at = e.m,
            Day4Action::Wake => sleeps.entry(id).or_insert(vec![]).push((sleep_at, e.m)),
        }
    }

    let a = sleeps
        .iter()
        .max_by_key::<usize, _>(|&(_, v)| v.iter().flat_map(|(f, t)| *f..*t).count())
        .map(|(k, v)| k * max_minute(v).1)
        .unwrap();

    let b = sleeps
        .iter()
        .map(|(k, v)| {
            let (f, m) = max_minute(v);
            (f, k * m)
        }).max()
        .unwrap()
        .1;

    (a, b)
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

    #[test]
    fn test_day4() {
        let input = read_input(4).unwrap();
        let (a, b) = day4(&input);
        assert_eq!(a, 19025);
        assert_eq!(b, 23776);
    }
}
