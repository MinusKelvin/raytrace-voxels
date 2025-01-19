use ahash::AHashMap;
use glam::IVec3;
use ordered_float::OrderedFloat;
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    pub struct Node;
}

pub struct SvoSpace {
    hash: AHashMap<SvoCell, Node>,
    cells: SlotMap<Node, RcSvoCell>,
    root: Option<Node>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum SvoCell {
    Block([OrderedFloat<f32>; 3]),
    Children([Option<Node>; 8]),
}

struct RcSvoCell {
    refcount: usize,
    cell: SvoCell,
    level: usize,
}

impl SvoSpace {
    pub fn new() -> Self {
        SvoSpace {
            hash: AHashMap::new(),
            cells: SlotMap::default(),
            root: None,
        }
    }

    pub fn set_root(&mut self, node: Option<Node>) {
        if let Some(node) = node {
            self.cells[node].refcount += 1;
        }
        if let Some(node) = self.root {
            self.decr(node);
        }
        self.root = node;
    }

    pub fn create_node(&mut self, cell: SvoCell) -> Option<Node> {
        if matches!(cell, SvoCell::Children(a) if a == [None; 8]) {
            return None;
        }
        let cells = &mut self.cells;
        Some(*self.hash.entry(cell).or_insert_with_key(|cell| {
            let mut level = 0;
            if let SvoCell::Children(children) = cell {
                for &child in children {
                    if let Some(child) = child {
                        cells[child].refcount += 1;
                        level = level.max(cells[child].level + 1);
                    }
                }
            }
            cells.insert(RcSvoCell {
                cell: cell.clone(),
                refcount: 0,
                level,
            })
        }))
    }

    fn decr(&mut self, node: Node) {
        let cell = &mut self.cells[node];
        cell.refcount -= 1;
        if cell.refcount == 0 {
            self.hash.remove(&cell.cell);
            let cell = self.cells.remove(node).unwrap();
            if let SvoCell::Children(children) = cell.cell {
                for child in children.into_iter().filter_map(|x| x) {
                    self.decr(child);
                }
            }
        }
    }

    pub fn get(&self, mut p: IVec3) -> Option<[f32; 3]> {
        let mut node = self.root;
        for level in (0..self.height()).rev() {
            let children = self.cells[node?].cell.unwrap_children();
            let c = IVec3::splat(1 << level);
            let above = p.cmpge(c);
            node = children[above.bitmask() as usize];
            p -= IVec3::select(above, c, IVec3::ZERO);
        }

        assert_eq!(p, IVec3::ZERO);
        let SvoCell::Block(color) = self.cells[node?].cell else {
            unreachable!()
        };

        Some(color.map(|v| v.0))
    }

    pub fn set(&mut self, mut p: IVec3, v: Option<[f32; 3]>) {
        let v = v.map(|a| a.map(OrderedFloat));

        let mut stack = vec![];
        let mut node = self.root;
        for level in (0..self.height()).rev() {
            let c = IVec3::splat(1 << level);
            let above = p.cmpge(c);
            stack.push((node, above.bitmask()));
            node =
                node.and_then(|n| self.cells[n].cell.unwrap_children()[above.bitmask() as usize]);
            p -= IVec3::select(above, c, IVec3::ZERO);
        }

        if let Some(n) = node {
            let SvoCell::Block(color) = self.cells[n].cell else {
                unreachable!()
            };
            if v == Some(color) {
                return;
            }
        } else if v.is_none() {
            return;
        }

        let mut new_node = v.and_then(|v| self.create_node(SvoCell::Block(v)));
        while let Some((n, idx)) = stack.pop() {
            let mut children = match n {
                Some(n) => *self.cells[n].cell.unwrap_children(),
                None => [None; 8],
            };
            children[idx as usize] = new_node;
            new_node = self.create_node(SvoCell::Children(children));
        }

        if let Some(n) = new_node {
            self.cells[n].refcount += 1;
        }
        if let Some(n) = self.root {
            self.decr(n);
        }
        self.root = new_node;
    }

    pub fn capacity(&self) -> usize {
        self.cells.capacity()
    }

    pub fn nodes(&self) -> impl Iterator<Item = (Node, &SvoCell)> {
        self.cells.iter().map(|(n, rc_cell)| (n, &rc_cell.cell))
    }

    pub fn get_node(&self, node: Node) -> &SvoCell {
        &self.cells[node].cell
    }

    pub fn root_node(&self) -> Option<Node> {
        self.root
    }

    pub fn height(&self) -> usize {
        self.root.map_or(0, |n| self.cells[n].level)
    }

    pub fn mem_usage(&self) -> usize {
        self.cells.capacity() * std::mem::size_of::<RcSvoCell>()
            + self.hash.capacity() * std::mem::size_of::<(SvoCell, Node)>()
    }
}

impl SvoCell {
    pub fn unwrap_children(&self) -> &[Option<Node>; 8] {
        match self {
            SvoCell::Block(_) => unreachable!(),
            SvoCell::Children(children) => children,
        }
    }
}
