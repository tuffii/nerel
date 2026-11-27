import unittest

from levels import assign_levels, layers_for_mode, nesting_depth


class LevelTests(unittest.TestCase):
    def test_nested_mentions_land_on_deeper_levels(self):
        entities = [
            (0, 20, "ORGANIZATION"),
            (0, 10, "CITY"),
            (2, 6, "PERSON"),
        ]
        self.assertEqual(nesting_depth(entities[0], entities), 0)
        self.assertEqual(nesting_depth(entities[1], entities), 1)
        self.assertEqual(nesting_depth(entities[2], entities), 2)
        levels, stats = assign_levels(entities)
        self.assertEqual(levels[0], [(0, 20, "ORGANIZATION")])
        self.assertEqual(levels[1], [(0, 10, "CITY")])
        self.assertEqual(levels[2], [(2, 6, "PERSON")])
        self.assertEqual(stats["selected_entities"], 3)
        self.assertEqual(stats["within_level_removed"], 0)

    def test_flat_mode_drops_nested_mentions(self):
        entities = [
            (0, 20, "ORGANIZATION"),
            (0, 10, "CITY"),
            (2, 6, "PERSON"),
        ]
        levels, stats = layers_for_mode(entities, "flat")
        self.assertEqual(levels[0], [(0, 20, "ORGANIZATION")])
        self.assertEqual(levels[1], [])
        self.assertEqual(levels[2], [])
        self.assertEqual(stats["selected_entities"], 1)

    def test_same_level_overlap_keeps_one_span(self):
        entities = [(0, 10, "EVENT"), (8, 14, "DATE")]
        levels, stats = assign_levels(entities)
        self.assertEqual(len(levels[0]), 1)
        self.assertEqual(stats["within_level_removed"], 1)
        self.assertEqual(levels[1], [])


if __name__ == "__main__":
    unittest.main()
