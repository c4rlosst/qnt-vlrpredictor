from valpredictor.models.veto import simulate_veto, trim_pool, veto_sequence

# P(team 1 wins) on each of seven maps: team 1 is best on Lotus/Ascent, team 2 on Haven/Split
P = {"Lotus": 0.70, "Ascent": 0.65, "Bind": 0.55, "Abyss": 0.50, "Sunset": 0.45, "Split": 0.35, "Haven": 0.30}


def test_veto_sequences():
    assert veto_sequence(3, 1) == [("ban", 1), ("ban", 2), ("pick", 1), ("pick", 2), ("ban", 1), ("ban", 2)]
    assert veto_sequence(3, 2)[0] == ("ban", 2)
    assert len(veto_sequence(1, 1)) == 6 and {a for a, _ in veto_sequence(1, 1)} == {"ban"}
    assert [a for a, _ in veto_sequence(5, 1)] == ["ban", "ban", "pick", "pick", "pick", "pick"]


def test_bo3_teams_ban_their_worst_and_pick_their_best():
    maps, picked_by = simulate_veto(P, 3, first_team=1)
    # team 1 bans Haven (its worst), team 2 bans Lotus (team 2's worst = team 1's best)
    # team 1 then picks Ascent (best left), team 2 picks Split (its best left)
    assert maps[0] == "Ascent" and picked_by["Ascent"] == 1
    assert maps[1] == "Split" and picked_by["Split"] == 2
    assert len(maps) == 3 and maps[2] not in picked_by          # decider has no picker
    assert "Haven" not in maps and "Lotus" not in maps          # both were banned


def test_first_ban_order_changes_the_maps():
    a, _ = simulate_veto(P, 3, first_team=1)
    b, _ = simulate_veto(P, 3, first_team=2)
    assert a != b


def test_bo1_and_bo5_lengths():
    assert len(simulate_veto(P, 1, 1)[0]) == 1
    maps5, picked5 = simulate_veto(P, 5, 1)
    assert len(maps5) == 5 and len(set(maps5)) == 5 and len(picked5) == 4


def test_small_pool_does_not_crash():
    maps, _ = simulate_veto({"Lotus": 0.6, "Bind": 0.5, "Haven": 0.4}, 3, 1)
    assert 1 <= len(maps) <= 3 and len(set(maps)) == len(maps)


def test_trim_pool_keeps_the_most_played_maps():
    weights = {f"m{i}": w for i, w in enumerate([0.2, 0.18, 0.15, 0.12, 0.1, 0.09, 0.08, 0.05, 0.03])}
    trimmed = trim_pool(weights, 7)
    assert len(trimmed) == 7 and "m8" not in trimmed and "m7" not in trimmed and "m0" in trimmed
