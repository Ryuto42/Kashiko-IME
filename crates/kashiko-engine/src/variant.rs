use std::collections::HashMap;
use std::sync::OnceLock;

const DATA: &str = include_str!("../data/variants.tsv");

fn table() -> &'static HashMap<char, Vec<char>> {
    static TABLE: OnceLock<HashMap<char, Vec<char>>> = OnceLock::new();
    TABLE.get_or_init(|| parse(DATA))
}

fn parse(data: &str) -> HashMap<char, Vec<char>> {
    let mut map = HashMap::new();
    for line in data.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, rest)) = line.split_once('\t') else {
            continue;
        };
        let mut chars = key.chars();
        let (Some(base), None) = (chars.next(), chars.next()) else {
            continue;
        };
        let vars: Vec<char> = rest
            .split_whitespace()
            .filter_map(|s| {
                let mut c = s.chars();
                match (c.next(), c.next()) {
                    (Some(v), None) if v != base => Some(v),
                    _ => None,
                }
            })
            .collect();
        if !vars.is_empty() {
            map.insert(base, vars);
        }
    }
    map
}

pub fn variants(text: &str, limit: usize) -> Vec<String> {
    let table = table();
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let push = |s: String, out: &mut Vec<String>| {
        if s != text && !out.contains(&s) {
            out.push(s);
        }
    };

    for (i, c) in chars.iter().enumerate() {
        let Some(vars) = table.get(c) else { continue };
        for v in vars {
            if out.len() >= limit {
                return out;
            }
            let s: String = chars
                .iter()
                .enumerate()
                .map(|(j, &x)| if j == i { *v } else { x })
                .collect();
            push(s, &mut out);
        }
    }

    let replaceable = chars.iter().filter(|c| table.contains_key(c)).count();
    if replaceable >= 2 {
        let max_rank = chars
            .iter()
            .filter_map(|c| table.get(c).map(Vec::len))
            .max()
            .unwrap_or(0);
        for rank in 0..max_rank {
            if out.len() >= limit {
                break;
            }
            let s: String = chars
                .iter()
                .map(|c| {
                    table
                        .get(c)
                        .and_then(|v| v.get(rank))
                        .copied()
                        .unwrap_or(*c)
                })
                .collect();
            push(s, &mut out);
        }
    }

    out.truncate(limit);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takahashi() {
        assert!(variants("高橋", 10).contains(&"髙橋".to_string()));
    }

    #[test]
    fn yoshida() {
        assert!(variants("吉田", 10).contains(&"𠮷田".to_string()));
    }

    #[test]
    fn saito() {
        let v = variants("斉藤", 10);
        assert!(v.contains(&"齊藤".to_string()));
        assert!(v.contains(&"齋藤".to_string()));
    }

    #[test]
    fn hamada() {
        let v = variants("浜田", 10);
        assert!(v.contains(&"濱田".to_string()));
        assert!(v.contains(&"濵田".to_string()));
    }

    #[test]
    fn combined_replacement() {
        let v = variants("斉高", 10);
        assert!(v.contains(&"齊髙".to_string()));
        assert!(!v.contains(&"斉高".to_string()));
    }

    #[test]
    fn no_target_is_empty() {
        assert!(variants("田中", 10).is_empty());
        assert!(variants("", 10).is_empty());
    }

    #[test]
    fn no_duplicates() {
        let v = variants("斉斉", 20);
        let mut d = v.clone();
        d.sort();
        d.dedup();
        assert_eq!(v.len(), d.len());
    }

    #[test]
    fn limit_applies() {
        assert_eq!(variants("崎", 2).len(), 2);
        assert_eq!(variants("崎", 0).len(), 0);
        assert_eq!(variants("崎", 100).len(), 3);
    }

    #[test]
    fn no_compat_ideographs() {
        const ALLOWED: [u32; 12] = [
            0xFA0E, 0xFA0F, 0xFA11, 0xFA13, 0xFA14, 0xFA1F, 0xFA21, 0xFA23, 0xFA24, 0xFA27, 0xFA28,
            0xFA29,
        ];
        for line in DATA.lines().filter(|l| !l.starts_with('#')) {
            for c in line.chars() {
                let u = c as u32;
                assert!(
                    !(0xF900..=0xFAFF).contains(&u) || ALLOWED.contains(&u),
                    "互換漢字: {c} U+{u:04X}"
                );
            }
        }
    }

    #[test]
    fn data_parses() {
        let data_lines = DATA
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .count();
        assert_eq!(table().len(), data_lines);
        assert!(data_lines >= 31);
        for (k, v) in table() {
            assert!(!v.contains(k));
        }
    }
}
