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
        10 => {
            let (a, b) = day10::run(&input);
            format!("\n{}{}", a, b)
        }
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
            })
            .unwrap();

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
        fn points(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
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
            })
            .collect();

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

    fn max_minute(input: &[(u32, u32)]) -> (u32, u32) {
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
            })
            .collect();

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
            })
            .max()
            .unwrap()
            .1;

        (a, b)
    }
}

mod day5 {
    fn poly_reduce(cs: &mut Vec<u8>) {
        let mut i = 0;
        while i < cs.len() - 1 {
            i += 1;
            let (a, b) = (cs[i - 1], cs[i]);
            if ((a as i8) - (b as i8)).abs() == 32 {
                cs.remove(i - 1);
                cs.remove(i - 1);
                i = i.saturating_sub(2);
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
            })
            .min()
            .unwrap();
        (a, b)
    }
}

mod day6 {
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};
    use std::i32::{MAX, MIN};

    fn manhattan(a: (i32, i32), b: (i32, i32)) -> i32 {
        (a.0 - b.0).abs() as i32 + (a.1 - b.1).abs() as i32
    }

    fn closest_point(points: &[(i32, i32)], coord: (i32, i32)) -> Option<(i32, i32)> {
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
            })
            .collect();

        let (min_x, max_x, min_y, max_y) = points.iter().fold((MAX, MIN, MAX, MIN), |acc, p| {
            (
                min(acc.0, p.0),
                max(acc.1, p.0),
                min(acc.2, p.1),
                max(acc.3, p.1),
            )
        });

        let mut map = HashMap::new();
        let mut edge = HashSet::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if let Some(p) = closest_point(&points, (x, y)) {
                    *map.entry(p).or_insert(0) += 1;
                    if p.0 == min_x || p.0 == max_x || p.1 == min_y || p.1 == max_y {
                        edge.insert(p.to_owned());
                    }
                }
            }
        }

        let a = map
            .iter()
            .filter_map(|(k, v)| if edge.contains(&k) { None } else { Some(v) })
            .max()
            .unwrap();

        let b = (min_x..=max_x)
            .flat_map(move |x| (min_y..=max_y).map(move |y| (x, y)))
            .filter(|a| points.iter().map(|b| manhattan(*a, *b)).sum::<i32>() < 10000)
            .count();
        (*a, b)
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
            })
            .next()
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
                })
                .collect();

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
        fn parse(data: &[usize], offs: &mut usize) -> Self {
            let n_children = data[*offs];
            let n_metadata: usize = data[*offs + 1];
            *offs += 2;
            Node {
                children: (0..n_children).map(|_| Node::parse(data, offs)).collect(),
                metadata: (0..n_metadata)
                    .map(|_| {
                        *offs += 1;
                        data[*offs - 1]
                    })
                    .collect(),
            }
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
        let root = Node::parse(&data, &mut 0);
        let a = root.checksum();
        let b = root.value();
        (a, b)
    }
}

mod day9 {
    use std::collections::{HashMap, LinkedList};

    fn game(players: usize, stop_at: usize) -> usize {
        let mut scores: HashMap<usize, usize> = HashMap::new();
        let mut circle: LinkedList<usize> = LinkedList::new();
        circle.push_back(0);
        circle.push_front(1);
        for marble in 2..=stop_at {
            if marble % 23 == 0 {
                let mut head = circle.split_off(circle.len() - 7);
                head.append(&mut circle);
                circle = head;
                *scores.entry(marble % players).or_insert(0) +=
                    marble + circle.pop_front().unwrap();
            } else {
                let mut head = circle.split_off(2);
                head.append(&mut circle);
                circle = head;
                circle.push_front(marble);
            }
        }

        *scores.values().max().unwrap()
    }

    pub fn run(input: &str) -> (usize, usize) {
        let words: Vec<_> = input.split(" ").collect();
        let players: usize = words[0].parse().unwrap();
        let stop_at: usize = words[6].parse().unwrap();

        let a = game(players, stop_at);
        let b = game(players, 100 * stop_at);

        (a, b)
    }
}

mod day10 {
    use regex::Regex;
    use std::i32::MAX;

    struct Star {
        x: i32,
        y: i32,
        dx: i32,
        dy: i32,
    }

    impl Star {
        fn pos_at(&self, tick: i32) -> (i32, i32) {
            (self.x + tick * self.dx, self.y + tick * self.dy)
        }
    }

    fn show_at(stars: &[Star], tick: i32) -> String {
        let points: Vec<_> = stars.iter().map(|s| s.pos_at(tick)).collect();
        let min_x = points.iter().map(|p| p.0).min().unwrap();
        let max_x = points.iter().map(|p| p.0).max().unwrap();
        let min_y = points.iter().map(|p| p.1).min().unwrap();
        let max_y = points.iter().map(|p| p.1).max().unwrap();

        let mut line = String::new();
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if points.iter().any(|p| p.0 == x && p.1 == y) {
                    line.push('#');
                } else {
                    line.push('.');
                }
            }
            line.push('\n');
        }

        line
    }

    pub fn run(input: &str) -> (String, i32) {
        let re = Regex::new(
            r"(?m)^position=<[ ]?([-]?\d+), +([-]?\d+)> velocity=<[ ]?([-]?\d+), +([-]?\d+)>$",
        )
        .unwrap();
        let stars: Vec<_> = re
            .captures_iter(input)
            .map(|m| Star {
                x: m[1].parse().unwrap(),
                y: m[2].parse().unwrap(),
                dx: m[3].parse().unwrap(),
                dy: m[4].parse().unwrap(),
            })
            .collect();

        let mut min_width = MAX;
        let mut tick = 0;
        loop {
            let points: Vec<_> = stars.iter().map(|s| s.pos_at(tick)).collect();
            let min_x = points.iter().map(|p| p.0).min().unwrap();
            let max_x = points.iter().map(|p| p.0).max().unwrap();
            let dx = max_x - min_x;
            if dx > min_width {
                tick -= 1;
                break;
            }
            min_width = dx;
            tick += 1;
        }
        let ans = show_at(&stars, tick);

        (ans, tick)
    }
}

mod day11 {
    use std::cmp::max;

    struct Grid {
        rows: Vec<Vec<i32>>,
        cols: Vec<Vec<i32>>,
    }

    impl Grid {
        fn new(serial: i32, size: usize) -> Self {
            let rows: Vec<Vec<i32>> = (0..size)
                .map(|y| {
                    (0..size)
                        .map(|x| {
                            let rack_id = 11 + x as i32;
                            (((rack_id * (y as i32 + 1) + serial) * rack_id / 100) % 10) - 5
                        })
                        .collect()
                })
                .collect();

            let cols: Vec<Vec<i32>> = (0..size)
                .map(|x| (0..size).map(|y| rows[y][x]).collect())
                .collect();
            Grid {
                rows: rows,
                cols: cols,
            }
        }

        fn score(&self, x: usize, y: usize, s: usize) -> i32 {
            self.box_scores(x, y).nth(s - 1).unwrap().1
        }

        fn box_scores(&self, x: usize, y: usize) -> BoxIter {
            BoxIter {
                matrix: self,
                score: 0,
                x: x,
                y: y,
                s: 0,
            }
        }

        fn len(&self) -> usize {
            self.rows.len()
        }
    }

    struct BoxIter<'a> {
        matrix: &'a Grid,
        score: i32,
        x: usize,
        y: usize,
        s: usize,
    }

    impl<'a> Iterator for BoxIter<'a> {
        type Item = (usize, i32);

        fn next(&mut self) -> Option<(usize, i32)> {
            if max(self.x, self.y) + self.s >= self.matrix.len() {
                None
            } else {
                self.score += self.matrix.rows[self.y + self.s][self.x..=self.x + self.s]
                    .iter()
                    .sum::<i32>();
                self.score += self.matrix.cols[self.x + self.s][self.y..self.y + self.s]
                    .iter()
                    .sum::<i32>();
                self.s += 1;
                Some((self.s, self.score))
            }
        }
    }

    pub fn run(input: &str) -> ((usize, usize), (usize, usize, usize)) {
        let serial: i32 = input.trim().parse().unwrap();
        let n = 300;
        let matrix = Grid::new(serial, n);

        let a = (0..n - 2)
            .flat_map(move |x| (0..n - 2).map(move |y| (x, y)))
            .max_by_key::<i32, _>(|(x, y)| matrix.score(*x, *y, 3))
            .map(|(x, y)| (x + 1, y + 1))
            .unwrap();

        let b = (0..n)
            .flat_map(move |x| (0..n).map(move |y| (x, y)))
            .map(|(x, y)| {
                let (size, score) = matrix.box_scores(x, y).max_by_key(|(_, s)| *s).unwrap();
                ((x, y, size), score)
            })
            .max_by_key::<i32, _>(|(_, score)| *score)
            .map(|((x, y, size), _)| (x + 1, y + 1, size))
            .unwrap();

        (a, b)
    }
}

mod day12 {
    use std::collections::HashMap;

    struct Pots {
        state: Vec<char>,
        ts: HashMap<Vec<char>, char>,
        start: i32,
        tick: usize,
    }

    impl Pots {
        fn pot_at(&self, pos: i32) -> char {
            *self.state.get((pos - self.start) as usize).unwrap_or(&'.')
        }

        fn state_at(&self, pos: i32) -> Vec<char> {
            (pos - 2..=pos + 2).map(|p| self.pot_at(p)).collect()
        }

        fn step(&mut self) {
            let new_state: Vec<char> = (-2..(self.state.len() as i32) + 2)
                .map(|i| {
                    *self
                        .ts
                        .get(&self.state_at(i as i32 + self.start))
                        .unwrap_or(&'.')
                })
                .collect();
            self.state = new_state;
            self.start -= 2;
            while self.state[0..=2] == ['.', '.', '.'] {
                self.state.remove(0);
                self.start += 1;
            }
            while self.state[self.state.len() - 3..self.state.len()] == ['.', '.', '.'] {
                self.state.pop();
            }
            self.tick += 1;
        }

        fn checksum(&self) -> i32 {
            (0..self.state.len() as i32)
                .map(|i| {
                    let pos = i + self.start;
                    (pos, self.pot_at(pos))
                })
                .filter(|(_, p)| *p == '#')
                .map(|(i, _)| i)
                .sum()
        }

        fn pattern(&self) -> String {
            format!("{}", self.state.iter().collect::<String>())
        }
    }

    pub fn run(input: &str) -> (i32, i64) {
        let mut lines = input.lines();
        let state: Vec<_> = lines
            .next()
            .unwrap()
            .rsplit(' ')
            .next()
            .unwrap()
            .chars()
            .collect();
        lines.next();

        let mut ts: HashMap<Vec<char>, char> = HashMap::new();
        loop {
            if let Some(l) = lines.next() {
                let cs: Vec<_> = l.chars().collect();
                let t1 = cs[0..5].to_vec();
                let t2 = cs[9];
                ts.insert(t1, t2);
            } else {
                break;
            }
        }

        let mut pots = Pots {
            state: state,
            ts: ts,
            start: 0,
            tick: 0,
        };

        for _ in 0..20 {
            pots.step();
        }
        let a = pots.checksum();

        let mut last: String = String::new();
        loop {
            pots.step();
            let next = pots.pattern();
            if next == last {
                break;
            }
            last = next;
        }
        let c1 = pots.checksum() as i64;
        pots.step();
        let c2 = pots.checksum() as i64;
        let dc = c2 - c1;

        let b = c2 + dc * (50000000000 - pots.tick as i64);

        (a, b)
    }
}

mod day13 {
    use std::collections::BTreeMap;

    struct Cart {
        dir: u8,
        state: u8,
        crashed: bool,
    }

    impl Cart {
        fn next_pos(&self, y: usize, x: usize) -> (usize, usize) {
            match self.dir {
                0 => (y - 1, x),
                1 => (y, x + 1),
                2 => (y + 1, x),
                3 => (y, x - 1),
                _ => unreachable!(),
            }
        }

        fn orient(&mut self, track: char) {
            match track {
                '/' => self.dir ^= 0b01,
                '\\' => self.dir ^= 0b11,
                '+' => {
                    self.dir = (self.dir
                        + match self.state {
                            0 => 3,
                            2 => 1,
                            _ => 0,
                        })
                        % 4;
                    self.state = (self.state + 1) % 3;
                }
                _ => (),
            }
        }
    }

    fn step(track: &Vec<Vec<char>>, carts: &mut BTreeMap<(usize, usize), Cart>) {
        for (y, x) in carts.keys().cloned().collect::<Vec<(usize, usize)>>() {
            let mut c = carts.remove(&(y, x)).unwrap();
            if !c.crashed {
                let new_yx = c.next_pos(y, x);
                if carts.contains_key(&new_yx) {
                    c.crashed = true;
                } else {
                    c.orient(track[new_yx.0][new_yx.1]);
                }
                carts.insert(new_yx, c);
            } else {
                carts.insert((y, x), c);
            }
        }
    }

    pub fn run(input: &str) -> ((usize, usize), (usize, usize)) {
        let mut track: Vec<Vec<char>> = input.lines().map(|l| l.chars().collect()).collect();

        let mut carts: BTreeMap<(usize, usize), Cart> = BTreeMap::new();
        for y in 0..track.len() {
            for x in 0..track[y].len() {
                if let Some(dir) = match track[y][x] {
                    '^' => Some(0),
                    '>' => Some(1),
                    'v' => Some(2),
                    '<' => Some(3),
                    _ => None,
                } {
                    track[y][x] = if dir % 2 == 0 { '|' } else { '-' };
                    carts.insert(
                        (y, x),
                        Cart {
                            dir: dir,
                            state: 0,
                            crashed: false,
                        },
                    );
                }
            }
        }

        while !carts.values().any(|c| c.crashed) {
            step(&track, &mut carts);
        }

        let crashed = carts.iter().find(|(_, c)| c.crashed).unwrap().0;
        let a = (crashed.1, crashed.0);

        while carts.values().filter(|c| !c.crashed).count() > 1 {
            carts = carts.into_iter().filter(|(_, c)| !c.crashed).collect();
            step(&track, &mut carts);
        }

        let alive = carts.iter().find(|(_, c)| !c.crashed).unwrap().0;
        let b = (alive.1, alive.0);

        (a, b)
    }
}

mod day14 {
    struct Recipes {
        rs: Vec<usize>,
        e1: usize,
        e2: usize,
        len: usize,
    }

    impl Recipes {
        fn new() -> Self {
            Recipes {
                rs: vec![3, 7],
                e1: 0,
                e2: 1,
                len: 2,
            }
        }

        fn ensure(&mut self, n: usize) {
            while self.len < n {
                let s = self.rs[self.e1] + self.rs[self.e2];
                if s > 9 {
                    self.rs.push(s / 10);
                    self.rs.push(s % 10);
                    self.len += 2;
                } else {
                    self.rs.push(s);
                    self.len += 1;
                }
                self.e1 = (self.e1 + self.rs[self.e1] + 1) % self.len;
                self.e2 = (self.e2 + self.rs[self.e2] + 1) % self.len;
            }
        }
    }

    pub fn run(input: &str) -> (usize, usize) {
        let n = input.trim().parse::<usize>().unwrap();
        let mut rs = Recipes::new();

        rs.ensure(10 + n);
        let a = rs.rs.iter().skip(n).take(10).fold(0, |acc, x| acc * 10 + x);

        let compare: Vec<_> = input
            .trim()
            .chars()
            .map(|c| c.to_digit(10).unwrap() as usize)
            .collect();

        let n2 = compare.len();
        let mut rs = Recipes::new();
        rs.ensure(n2);
        let mut i = 0;
        while *compare != rs.rs[i..i + n2] {
            i += 1;
            rs.ensure(i + n2);
        }

        (a, i)
    }
}

mod day15 {
    use std::collections::{BTreeMap, BTreeSet, HashSet};
    use std::fmt;

    type Pos = (usize, usize);

    fn adjacent(pos: &Pos) -> impl Iterator<Item = Pos> + '_ {
        let (y, x) = pos;
        static ADJ: &[(isize, isize)] = &[(-1, 0), (0, -1), (0, 1), (1, 0)];
        ADJ.iter()
            .map(move |(dy, dx)| ((*y as isize + dy) as usize, (*x as isize + dx) as usize))
    }

    struct Unit {
        race: char,
        hp: u8,
    }

    impl Unit {
        fn new(race: char) -> Self {
            Unit {
                race: race,
                hp: 200,
            }
        }
    }

    struct Cave {
        map: Vec<Vec<char>>,
        units: BTreeMap<Pos, Unit>,
    }

    impl Cave {
        fn parse(input: &str) -> Self {
            let mut map: Vec<Vec<char>> = input.lines().map(|l| l.chars().collect()).collect();
            let mut units: BTreeMap<Pos, Unit> = BTreeMap::new();
            for y in 0..map.len() {
                for x in 0..map[y].len() {
                    if let Some(race) = match map[y][x] {
                        'G' => Some('G'),
                        'E' => Some('E'),
                        _ => None,
                    } {
                        map[y][x] = '.';
                        units.insert((y, x), Unit::new(race));
                    }
                }
            }

            Cave {
                map: map,
                units: units,
            }
        }

        fn find_move(&self, start: Pos, race: char) -> Pos {
            let targets: HashSet<_> = self
                .units
                .iter()
                .filter(move |(_, u)| u.race != race)
                .flat_map(|(p, _)| {
                    adjacent(p)
                        .filter(|p| !self.units.contains_key(&p) && self.map[p.0][p.1] == '.')
                })
                .collect();
            if targets.is_empty() {
                return start;
            }
            let mut explored = HashSet::new();
            let mut unexplored: Vec<Vec<Pos>> = vec![vec![start]];
            let mut found = false;
            let mut distance = 0;
            let mut shortest: BTreeSet<Vec<Pos>> = BTreeSet::new();

            while unexplored.len() > 0 {
                let path = unexplored.remove(0);
                let p = *path.last().unwrap();
                if explored.insert(p) {
                    if targets.contains(&p) {
                        if !found {
                            found = true;
                            distance = path.len();
                        }
                        if path.len() == distance {
                            let mut a_path = path.clone();
                            a_path[0] = p;
                            shortest.insert(a_path);
                        }
                    } else if !found && !self.units.contains_key(&p) && self.map[p.0][p.1] == '.' {
                        for a in adjacent(&p) {
                            let mut a_path = path.clone();
                            a_path.push(a);
                            unexplored.push(a_path);
                        }
                    }
                }
            }

            if let Some(path) = shortest.iter().next() {
                if path.len() >= 2 {
                    return path[1];
                }
            }

            start
        }

        fn target(&self, pos: Pos, race: char) -> Option<Pos> {
            let mut target = None;
            let mut low_hp = 201;
            for p in adjacent(&pos) {
                if let Some(o) = self.units.get(&p) {
                    if o.race != race {
                        if o.hp < low_hp {
                            target = Some(p);
                            low_hp = o.hp;
                        }
                    }
                }
            }
            target
        }

        fn step(&mut self, e_attack: u8) -> bool {
            let mut moved: HashSet<Pos> = HashSet::new();
            let mut over = false;
            for pos in self.units.keys().cloned().collect::<Vec<(usize, usize)>>() {
                if moved.insert(pos) {
                    if over {
                        return false;
                    }
                    if let Some(unit) = self.units.remove(&pos) {
                        let new_pos = self.find_move(pos, unit.race);
                        if let Some(target) = self.target(new_pos, unit.race) {
                            let mut enemy = self.units.remove(&target).unwrap();
                            let dmg = if enemy.race == 'E' { 3 } else { e_attack };
                            enemy.hp = enemy.hp.saturating_sub(dmg);
                            if enemy.hp > 0 {
                                self.units.insert(target, enemy);
                            } else {
                                moved.insert(target);
                                over = self
                                    .units
                                    .values()
                                    .map(|u| u.race)
                                    .collect::<HashSet<_>>()
                                    .len()
                                    < 2;
                            }
                        }
                        self.units.insert(new_pos, unit);
                    }
                }
            }
            true
        }

        fn n_elves(&self) -> usize {
            self.units.values().filter(|u| u.race == 'E').count()
        }
    }

    impl fmt::Debug for Cave {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            for y in 0..self.map.len() {
                let mut row = String::new();
                for x in 0..self.map[y].len() {
                    row.push(if let Some(unit) = self.units.get(&(y, x)) {
                        unit.race
                    } else {
                        self.map[y][x]
                    });
                }
                write!(f, "{}\n", row)?
            }
            Ok(())
        }
    }

    pub fn run(input: &str) -> (usize, usize) {
        let mut cave = Cave::parse(input);
        let n = cave.n_elves();
        //println!("{:?}", cave);
        let mut tick = 0;
        while cave.step(3) {
            tick += 1;
        }
        //println!("{:?}", cave);
        let a = tick * cave.units.values().map(|u| u.hp as usize).sum::<usize>();

        let mut attack = 3;
        let b = loop {
            let mut cave2 = Cave::parse(input);
            tick = 0;
            attack += 1;
            while cave2.n_elves() == n && cave2.step(attack) {
                tick += 1;
            }
            if cave2.n_elves() == n {
                break tick * cave2.units.values().map(|u| u.hp as usize).sum::<usize>();
            }
        };

        (a, b)
    }
}

mod day16 {
    use self::Op::*;
    use self::Read::*;
    use std::collections::HashSet;

    #[derive(Debug, PartialEq, Eq, Hash)]
    enum Read {
        Reg,
        Imm,
    }

    impl Read {
        fn get(&self, val: usize, regs: &[i32]) -> i32 {
            match self {
                Reg => regs[val],
                Imm => val as i32,
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    enum Op {
        Add(Read),
        Mul(Read),
        Ban(Read),
        Bor(Read),
        Set(Read),
        Gtt(Read, Read),
        Equ(Read, Read),
    }

    static OPS: &'static [Op] = &[
        Add(Reg),
        Add(Imm),
        Mul(Reg),
        Mul(Imm),
        Ban(Reg),
        Ban(Imm),
        Bor(Reg),
        Bor(Imm),
        Set(Reg),
        Set(Imm),
        Gtt(Imm, Reg),
        Gtt(Reg, Imm),
        Gtt(Reg, Reg),
        Equ(Imm, Reg),
        Equ(Reg, Imm),
        Equ(Reg, Reg),
    ];

    impl Op {
        fn exec(&self, a: usize, b: usize, c: usize, regs: &Vec<i32>) -> Vec<i32> {
            let mut ret = regs.clone();
            match self {
                Add(read) => ret[c] = Reg.get(a, regs) + read.get(b, regs),
                Mul(read) => ret[c] = Reg.get(a, regs) * read.get(b, regs),
                Ban(read) => ret[c] = Reg.get(a, regs) & read.get(b, regs),
                Bor(read) => ret[c] = Reg.get(a, regs) | read.get(b, regs),
                Set(read) => ret[c] = read.get(a, regs),
                Gtt(read_a, read_b) => {
                    ret[c] = if read_a.get(a, regs) > read_b.get(b, regs) {
                        1
                    } else {
                        0
                    }
                }
                Equ(read_a, read_b) => {
                    ret[c] = if read_a.get(a, regs) == read_b.get(b, regs) {
                        1
                    } else {
                        0
                    }
                }
            }
            ret
        }
    }

    pub fn run(input: &str) -> (usize, i32) {
        let mut a = 0;
        let mut codes: Vec<HashSet<&Op>> = (0..16).map(|_| OPS.iter().collect()).collect();
        let mut lines = input.lines();
        loop {
            let regs = lines.next().unwrap();
            if regs == "" {
                break;
            }
            let regs: Vec<i32> = regs
                .split_at(9)
                .1
                .split_at(10)
                .0
                .split(", ")
                .map(|x| x.parse().unwrap())
                .collect();
            let args: Vec<usize> = lines
                .next()
                .unwrap()
                .split(' ')
                .map(|x| x.parse().unwrap())
                .collect();
            let out: Vec<i32> = lines
                .next()
                .unwrap()
                .split_at(9)
                .1
                .split_at(10)
                .0
                .split(", ")
                .map(|x| x.parse().unwrap())
                .collect();
            lines.next();
            if OPS
                .iter()
                .filter(|op| op.exec(args[1], args[2], args[3], &regs) == out)
                .count()
                >= 3
            {
                a += 1;
            }
            let new_codes: HashSet<&Op> = codes[args[0]]
                .iter()
                .cloned()
                .filter(|op| op.exec(args[1], args[2], args[3], &regs) == out)
                .collect();
            codes[args[0]] = new_codes;
        }

        lines.next();

        let mut old_state: Vec<_> = codes.iter().map(|ops| ops.len()).collect();
        let mut new_state: Vec<usize> = vec![];
        while old_state != new_state {
            for op in OPS {
                if codes.iter().any(|ops| ops.len() == 1 && ops.contains(op)) {
                    for ops in codes.iter_mut() {
                        if ops.len() > 1 && ops.contains(op) {
                            ops.remove(op);
                        }
                    }
                }
            }

            old_state = new_state;
            new_state = codes.iter().map(|ops| ops.len()).collect();
        }
        let codes: Vec<_> = codes.iter().map(|ops| ops.iter().next().unwrap()).collect();
        let mut regs = vec![0, 0, 0, 0];
        while let Some(l) = lines.next() {
            let args: Vec<usize> = l.split(' ').map(|x| x.parse().unwrap()).collect();
            regs = codes[args[0]].exec(args[1], args[2], args[3], &regs)
        }
        let b = regs[0];

        (a, b)
    }
}

mod day17 {
    use std::cmp::{max, min};
    use std::collections::HashSet;
    use std::fmt;
    use std::usize::MAX;

    type Pos = (usize, usize);

    struct Map {
        clay: HashSet<Pos>,
        running: HashSet<Pos>,
        still: HashSet<Pos>,
        min_y: usize,
        max_y: usize,
    }

    impl Map {
        fn parse(input: &str) -> Self {
            let mut clay: HashSet<Pos> = HashSet::new();
            let mut min_y = MAX;
            let mut max_y = 0;
            for line in input.lines() {
                let v = line.starts_with("x");
                let mut parts = line.split(", ");
                let a: usize = parts.next().unwrap()[2..].parse().unwrap();
                let mut range = parts.next().unwrap()[2..].split("..");
                let b: usize = range.next().unwrap().parse().unwrap();
                let c: usize = range.next().unwrap().parse().unwrap();
                for i in b..=c {
                    if v {
                        clay.insert((a, i));
                        min_y = min(min_y, i);
                        max_y = max(max_y, i);
                    } else {
                        clay.insert((i, a));
                        min_y = min(min_y, a);
                        max_y = max(max_y, a);
                    }
                }
            }
            let mut running = HashSet::new();
            running.insert((500, 0));
            Map {
                clay: clay,
                running: running,
                still: HashSet::new(),
                min_y: min_y,
                max_y: max_y,
            }
        }

        fn empty(&self, pos: Pos) -> bool {
            !self.clay.contains(&pos) && !self.still.contains(&pos) && !self.running.contains(&pos)
        }

        fn go(&mut self) {
            let mut active = vec![(500, 1)];
            while let Some((x, y)) = active.pop() {
                let mut x = x;
                let mut y = y;
                loop {
                    self.running.insert((x, y));
                    if !self.clay.contains(&(x, y + 1)) && !self.still.contains(&(x, y + 1)) {
                        if y < self.max_y {
                            y += 1;
                        } else {
                            break;
                        }
                    } else if self.empty((x + 1, y)) {
                        if !self.running.contains(&(x - 1, y)) {
                            active.push((x, y));
                        }
                        x += 1;
                    } else if self.empty((x - 1, y)) {
                        x -= 1;
                    } else if self.clay.contains(&(x - 1, y)) {
                        let mut w = x;
                        while self.running.contains(&(w + 1, y)) {
                            w += 1;
                        }
                        if self.clay.contains(&(w + 1, y)) {
                            for i in x..=w {
                                self.running.remove(&(i, y));
                                self.still.insert((i, y));
                                if self.running.contains(&(i, y - 1)) {
                                    active.push((i, y - 1));
                                }
                            }
                        }
                        break;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    impl fmt::Debug for Map {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut min_x = MAX;
            let mut max_x = 0;
            for &(x, _) in &self.clay {
                min_x = min(min_x, x);
                max_x = max(max_x, x);
            }
            for y in self.min_y..=self.max_y {
                let mut row = String::new();
                for x in (min_x - 1)..=(max_x + 1) {
                    let p = (x, y);
                    if self.clay.contains(&p) {
                        row.push('#');
                    } else if self.running.contains(&p) {
                        row.push('|');
                    } else if self.still.contains(&p) {
                        row.push('~');
                    } else {
                        row.push('.');
                    }
                }
                write!(f, "{}\n", row)?
            }
            Ok(())
        }
    }

    pub fn run(input: &str) -> (usize, usize) {
        let mut map = Map::parse(input);
        map.go();
        //println!("{:?}", map);

        let a = map.running.len() + map.still.len() - map.min_y;
        let b = map.still.len();
        (a, b)
    }
}

mod day18 {
    use std::cmp::min;
    use std::collections::HashSet;

    fn adjacent(map: &Vec<Vec<char>>, x: usize, y: usize) -> Vec<char> {
        let min_y = y.saturating_sub(1);
        let min_x = x.saturating_sub(1);
        let max_y = min(49, y + 1);
        let max_x = min(49, x + 1);

        (min_y..=max_y)
            .flat_map(|j| (min_x..=max_x).map(move |i| (i, j)))
            .filter(|&p| p != (x, y))
            .map(|(i, j)| map[j][i])
            .collect()
    }

    fn next_state(map: &Vec<Vec<char>>, x: usize, y: usize) -> char {
        let around = adjacent(map, x, y);
        match map[y][x] {
            '.' => {
                if around.iter().filter(|&x| x == &'|').count() >= 3 {
                    '|'
                } else {
                    '.'
                }
            }
            '|' => {
                if around.iter().filter(|&x| x == &'#').count() >= 3 {
                    '#'
                } else {
                    '|'
                }
            }
            '#' => {
                if around.iter().any(|&x| x == '#') && around.iter().any(|&x| x == '|') {
                    '#'
                } else {
                    '.'
                }
            }
            _ => '?',
        }
    }

    fn tick(map: &Vec<Vec<char>>) -> Vec<Vec<char>> {
        let mut next: Vec<Vec<char>> = vec![];
        for y in 0..50 {
            let mut row: Vec<char> = vec![];
            for x in 0..50 {
                row.push(next_state(map, x, y));
            }
            next.push(row);
        }
        next
    }

    fn value(map: &Vec<Vec<char>>) -> usize {
        let wood: usize = map
            .iter()
            .map(|row| row.iter().filter(|&x| x == &'|').count())
            .sum();
        let lumber: usize = map
            .iter()
            .map(|row| row.iter().filter(|&x| x == &'#').count())
            .sum();
        wood * lumber
    }

    pub fn run(input: &str) -> (usize, usize) {
        let mut map: Vec<Vec<char>> = input.lines().map(|l| l.chars().collect()).collect();
        let mut t = 10;
        for _ in 0..t {
            map = tick(&map);
        }
        let a = value(&map);

        let mut seen = HashSet::new();
        while seen.insert(map.clone()) {
            map = tick(&map);
            t += 1;
        }

        let start = t;
        let target = map.clone();
        map = tick(&map);
        t += 1;

        while map != target {
            map = tick(&map);
            t += 1;
        }

        let cycle = t - start;
        let remaining = (1000000000 - t) % cycle;
        for _ in 0..remaining {
            map = tick(&map);
        }
        let b = value(&map);

        (a, b)
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

    #[test]
    fn test_day9() {
        let input = read_input(9).unwrap();
        let (a, b) = day9::run(&input);
        assert_eq!(a, 374690);
        assert_eq!(b, 3009951158);
    }

    #[test]
    fn test_day10() {
        let input = read_input(10).unwrap();
        let (a, b) = day10::run(&input);
        assert_eq!(
            a,
            "\
             #.......#....#.....###..######..#....#....##....#....#....##..\n\
             #.......#....#......#...#.......#...#....#..#...#....#...#..#.\n\
             #........#..#.......#...#.......#..#....#....#...#..#...#....#\n\
             #........#..#.......#...#.......#.#.....#....#...#..#...#....#\n\
             #.........##........#...#####...##......#....#....##....#....#\n\
             #.........##........#...#.......##......######....##....######\n\
             #........#..#.......#...#.......#.#.....#....#...#..#...#....#\n\
             #........#..#...#...#...#.......#..#....#....#...#..#...#....#\n\
             #.......#....#..#...#...#.......#...#...#....#..#....#..#....#\n\
             ######..#....#...###....#.......#....#..#....#..#....#..#....#\n\
             "
        );
        assert_eq!(b, 10312);
    }

    #[test]
    fn test_day11() {
        let input = read_input(11).unwrap();
        let (a, b) = day11::run(&input);
        assert_eq!(a, (235, 22));
        assert_eq!(b, (231, 135, 8));
    }

    #[test]
    fn test_day12() {
        let input = read_input(12).unwrap();
        let (a, b) = day12::run(&input);
        assert_eq!(a, 1672);
        assert_eq!(b, 1650000000055);
    }

    #[test]
    fn test_day13() {
        let input = read_input(13).unwrap();
        let (a, b) = day13::run(&input);
        assert_eq!(a, (118, 66));
        assert_eq!(b, (70, 129));
    }

    #[test]
    fn test_day14() {
        let input = read_input(14).unwrap();
        let (a, b) = day14::run(&input);
        assert_eq!(a, 6126491027);
        assert_eq!(b, 20191616);
    }

    #[test]
    fn test_day15() {
        let input = read_input(15).unwrap();
        let (a, b) = day15::run(&input);
        assert_eq!(a, 220480);
        assert_eq!(b, 53576);
    }

    #[test]
    fn test_day16() {
        let input = read_input(16).unwrap();
        let (a, b) = day16::run(&input);
        assert_eq!(a, 651);
        assert_eq!(b, 706);
    }

    #[test]
    fn test_day17() {
        let input = read_input(17).unwrap();
        let (a, b) = day17::run(&input);
        assert_eq!(a, 34291);
        assert_eq!(b, 28487);
    }

    #[test]
    fn test_day18() {
        let input = read_input(18).unwrap();
        let (a, b) = day18::run(&input);
        assert_eq!(a, 574590);
        assert_eq!(b, 183787);
    }
}
