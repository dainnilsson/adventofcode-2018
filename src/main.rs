extern crate regex;

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
        1 => format!("{:?}", day1::run(&input)),
        2 => format!("{:?}", day2::run(&input)),
        3 => format!("{:?}", day3::run(&input)),
        4 => format!("{:?}", day4::run(&input)),
        5 => format!("{:?}", day5::run(&input)),
        6 => format!("{:?}", day6::run(&input)),
        7 => format!("{:?}", day7::run(&input)),
        8 => format!("{:?}", day8::run(&input)),
        9 => format!("{:?}", day9::run(&input)),
        10 => format!("{:?}", day10::run(&input)),
        11 => format!("{:?}", day11::run(&input)),
        12 => format!("{:?}", day12::run(&input)),
        13 => format!("{:?}", day13::run(&input)),
        14 => format!("{:?}", day14::run(&input)),
        15 => format!("{:?}", day15::run(&input)),
        16 => format!("{:?}", day16::run(&input)),
        17 => format!("{:?}", day17::run(&input)),
        18 => format!("{:?}", day18::run(&input)),
        19 => format!("{:?}", day19::run(&input)),
        20 => format!("{:?}", day20::run(&input)),
        21 => format!("{:?}", day21::run(&input)),
        22 => format!("{:?}", day22::run(&input)),
        23 => format!("{:?}", day23::run(&input)),
        24 => format!("{:?}", day24::run(&input)),
        _ => panic!("Day not implemented!"),
    })
}

fn read_input(i: u32) -> io::Result<String> {
    let mut contents = String::new();
    File::open(format!("input{}.txt", i))?.read_to_string(&mut contents)?;
    Ok(contents)
}

mod day1 {
    use std::collections::HashSet;

    pub fn run(input: &str) -> (i32, i32) {
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
}

mod day2 {
    fn duplicates(x: &str, i: usize) -> bool {
        x.chars()
            .any(|c| x.chars().filter(|&a| a == c).count() == i)
    }

    pub fn run(input: &str) -> (usize, String) {
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
}

mod day3 {
    use regex::Regex;
    use std::collections::HashSet;

    struct Claim {
        id: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    }

    impl Claim {
        fn points<'a>(&'a self) -> impl Iterator<Item = (u32, u32)> + 'a {
            (self.x..self.x + self.w)
                .flat_map(move |x| (self.y..self.y + self.h).map(move |y| (x, y)))
        }
    }

    pub fn run(input: &str) -> (usize, u32) {
        let re = Regex::new(r"(?m)^#(\d+) @ (\d+),(\d+): (\d+)x(\d+)").unwrap();
        let boxes: Vec<Claim> = re
            .captures_iter(input)
            .map(|m| Claim {
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
}

mod day4 {
    use regex::Regex;
    use std::collections::HashMap;

    #[derive(Ord, PartialOrd, PartialEq, Eq)]
    enum Action {
        Begin(u32),
        Sleep,
        Wake,
    }

    #[derive(Ord, PartialOrd, PartialEq, Eq)]
    struct Entry {
        year: u32,
        month: u32,
        day: u32,
        h: u32,
        m: u32,
        action: Action,
    }

    fn max_minute(input: &Vec<(u32, u32)>) -> (u32, u32) {
        let mut ms = HashMap::new();
        input
            .iter()
            .flat_map(|(f, t)| *f..*t)
            .for_each(|x| *ms.entry(x).or_insert(0) += 1);
        ms.iter()
            .max_by_key(|(_, f)| *f)
            .map(|(&m, &f)| (f, m))
            .unwrap()
    }

    pub fn run(input: &str) -> (u32, u32) {
        let re_entry =
            Regex::new(r"(?m)^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})\] (.+)$").unwrap();
        let re_id = Regex::new(r".*#(\d+) .*").unwrap();
        let mut entries: Vec<Entry> = re_entry
            .captures_iter(input)
            .map(|m| Entry {
                year: m[1].parse().unwrap(),
                month: m[2].parse().unwrap(),
                day: m[3].parse().unwrap(),
                h: m[4].parse().unwrap(),
                m: m[5].parse().unwrap(),
                action: match &m[6] {
                    "wakes up" => Action::Wake,
                    "falls asleep" => Action::Sleep,
                    _ => Action::Begin(re_id.captures(&m[6]).unwrap()[1].parse().unwrap()),
                },
            }).collect();

        entries.sort();

        let mut id = 0;
        let mut sleep_at = 0;
        let mut sleeps: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        for e in &entries {
            match e.action {
                Action::Begin(x) => id = x,
                Action::Sleep => sleep_at = e.m,
                Action::Wake => sleeps.entry(id).or_insert(vec![]).push((sleep_at, e.m)),
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
}

mod day5 {
    use std::cmp;

    fn poly_reduce(cs: &mut Vec<u8>) {
        let mut i = 0;
        while i < cs.len() - 1 {
            i += 1;
            let (a, b) = (cs[i - 1], cs[i]);
            if ((a as i8) - (b as i8)).abs() == 32 {
                cs.remove(i - 1);
                cs.remove(i - 1);
                i = cmp::max(0, i as i32 - 2) as usize;
            }
        }
    }

    pub fn run(input: &str) -> (usize, usize) {
        let mut cs: Vec<_> = input.trim().bytes().collect();
        poly_reduce(&mut cs);
        let a = cs.len();

        let b = (65u8..=90)
            .map(|i| {
                let mut x: Vec<_> = cs
                    .iter()
                    .filter(|&x| *x != i && *x != i + 32)
                    .map(|&x| x)
                    .collect();
                poly_reduce(&mut x);
                x.len()
            }).min()
            .unwrap();
        (a, b)
    }
}

mod day6 {
    use std::cmp;
    use std::collections::HashMap;

    fn manhattan(a: (i32, i32), b: (i32, i32)) -> i32 {
        (a.0 - b.0).abs() as i32 + (a.1 - b.1).abs() as i32
    }

    fn closest_point(points: &Vec<(i32, i32)>, coord: (i32, i32)) -> Option<(i32, i32)> {
        let mut closest = None;
        let mut distance = std::i32::MAX;
        for (x, y) in points {
            let dist = manhattan(coord, (*x, *y));
            if dist == distance {
                closest = None;
            } else if dist < distance {
                distance = dist;
                closest = Some((*x, *y));
            }
        }
        closest
    }

    pub fn run(input: &str) -> (i32, usize) {
        let points: Vec<(i32, i32)> = input
            .lines()
            .map(|l| {
                let mut parts = l.split(", ").map(|x| x.parse().unwrap());
                let x = parts.next().unwrap();
                let y = parts.next().unwrap();
                (x, y)
            }).collect();

        let (min_x, max_x, min_y, max_y) = points.iter().fold(
            (std::i32::MAX, std::i32::MIN, std::i32::MAX, std::i32::MIN),
            |acc, p| {
                (
                    cmp::min(acc.0, p.0),
                    cmp::max(acc.1, p.0),
                    cmp::min(acc.2, p.1),
                    cmp::max(acc.3, p.1),
                )
            },
        );

        let mut map = HashMap::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if let Some(p) = closest_point(&points, (x, y)) {
                    *map.entry(p).or_insert(0) += 1;
                };
            }
        }

        let mut largest = map.clone();
        for x in min_x..=max_x {
            for y in [min_y, max_y].iter() {
                if let Some(p) = closest_point(&points, (x, *y)) {
                    largest.remove(&p);
                }
            }
        }
        for y in min_y..=max_y {
            for x in [min_x, max_x].iter() {
                if let Some(p) = closest_point(&points, (*x, y)) {
                    largest.remove(&p);
                }
            }
        }

        let a = *largest.values().max().unwrap();

        let b = (min_x..=max_x)
            .flat_map(move |x| (min_y..max_y).map(move |y| (x, y)))
            .filter(|a| points.iter().map(|b| manhattan(*a, *b)).sum::<i32>() < 10000)
            .count();
        (a, b)
    }
}

mod day7 {
    use regex::Regex;
    use std::collections::{BTreeMap, HashSet};

    fn get_work(
        deps: &BTreeMap<char, Vec<char>>,
        done: &Vec<char>,
        excl: &HashSet<char>,
    ) -> Option<char> {
        deps.iter()
            .filter(|(k, v)| {
                !done.contains(k) && !excl.contains(k) && v.iter().all(|x| done.contains(&x))
            }).next()
            .map(|(&k, _)| k)
    }

    pub fn run(input: &str) -> (String, i32) {
        let re =
            Regex::new(r"(?m)^\Step (\w) must be finished before step (\w) can begin.$").unwrap();
        let mut deps: BTreeMap<char, Vec<char>> = BTreeMap::new();
        for m in re.captures_iter(input) {
            let a = m[2].chars().next().unwrap();
            let b = m[1].chars().next().unwrap();
            deps.entry(a).or_insert(vec![]).push(b);
            deps.entry(b).or_insert(vec![]);
        }

        let mut done: Vec<char> = vec![];
        while done.len() < deps.len() {
            let step = get_work(&deps, &done, &HashSet::new()).unwrap();
            done.push(step);
        }
        let a: String = done.iter().collect();

        let mut workers: Vec<_> = (0..5).map(|_| (0, '.')).collect();
        let mut done: Vec<char> = vec![];
        let mut doing: HashSet<char> = HashSet::new();
        let mut time = 0;

        while done.len() < deps.len() {
            workers = workers
                .iter()
                .map(|(t, c_done)| {
                    if *t <= time {
                        if let Some(c) = get_work(&deps, &done, &doing) {
                            doing.insert(c);
                            (time - 4 + (c as u8) as i32, c)
                        } else {
                            (*t, '.')
                        }
                    } else {
                        (*t, *c_done)
                    }
                }).collect();

            time = workers
                .iter()
                .map(|w| w.0)
                .filter(|&t| t > time)
                .min()
                .unwrap_or(time);

            workers
                .iter()
                .filter(|(t, _)| *t <= time)
                .for_each(|(_, c)| {
                    if doing.remove(c) {
                        done.push(*c);
                    }
                });
        }

        (a, time)
    }
}

mod day8 {
    struct Node {
        metadata: Vec<usize>,
        children: Vec<Node>,
    }

    impl Node {
        fn parse(data: &Vec<usize>, offs: usize) -> (Node, usize) {
            let mut offs = offs;
            let n_children = data[offs];
            let n_metadata: usize = data[offs + 1];
            offs += 2;
            let mut n = Node {
                metadata: vec![],
                children: vec![],
            };
            for _ in 0..n_children {
                let (child, n_offs) = Node::parse(data, offs);
                offs = n_offs;
                n.children.push(child);
            }
            n.metadata = data[offs..offs + n_metadata].to_vec();

            (n, offs + n_metadata)
        }

        fn checksum(&self) -> usize {
            self.children.iter().map(|c| c.checksum()).sum::<usize>()
                + self.metadata.iter().sum::<usize>()
        }

        fn value(&self) -> usize {
            if self.children.len() == 0 {
                self.checksum()
            } else {
                let c_vals: Vec<_> = self.children.iter().map(|c| c.value()).collect();
                self.metadata
                    .iter()
                    .filter(|&m| *m <= c_vals.len())
                    .map(|m| c_vals[m - 1])
                    .sum()
            }
        }
    }

    pub fn run(input: &str) -> (usize, usize) {
        let data: Vec<usize> = input
            .trim()
            .split(" ")
            .map(|c| c.parse().unwrap())
            .collect();
        let root = Node::parse(&data, 0).0;
        let a = root.checksum();
        let b = root.value();
        (a, b)
    }
}

mod day9 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day10 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day11 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day12 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day13 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day14 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day15 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day16 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day17 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day18 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day19 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day20 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day21 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day22 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day23 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

mod day24 {
    pub fn run(_input: &str) -> (usize, usize) {
        unimplemented!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_day1() {
        let input = read_input(1).unwrap();
        let (a, b) = day1::run(&input);
        assert_eq!(a, 502);
        assert_eq!(b, 71961);
    }

    #[test]
    fn test_day2() {
        let input = read_input(2).unwrap();
        let (a, b) = day2::run(&input);
        assert_eq!(a, 5976);
        assert_eq!(b, "xretqmmonskvzupalfiwhcfdb");
    }

    #[test]
    fn test_day3() {
        let input = read_input(3).unwrap();
        let (a, b) = day3::run(&input);
        assert_eq!(a, 101565);
        assert_eq!(b, 656);
    }

    #[test]
    fn test_day4() {
        let input = read_input(4).unwrap();
        let (a, b) = day4::run(&input);
        assert_eq!(a, 19025);
        assert_eq!(b, 23776);
    }

    #[test]
    fn test_day5() {
        let input = read_input(5).unwrap();
        let (a, b) = day5::run(&input);
        assert_eq!(a, 10496);
        assert_eq!(b, 5774);
    }

    #[test]
    fn test_day6() {
        let input = read_input(6).unwrap();
        let (a, b) = day6::run(&input);
        assert_eq!(a, 5333);
        assert_eq!(b, 35334);
    }

    #[test]
    fn test_day7() {
        let input = read_input(7).unwrap();
        let (a, b) = day7::run(&input);
        assert_eq!(a, "JKNSTHCBGRVDXWAYFOQLMPZIUE");
        assert_eq!(b, 755);
    }

    #[test]
    fn test_day8() {
        let input = read_input(8).unwrap();
        let (a, b) = day8::run(&input);
        assert_eq!(a, 37905);
        assert_eq!(b, 33891);
    }
}
