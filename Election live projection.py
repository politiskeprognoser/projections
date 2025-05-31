import time
start_time = time.time()
import pandas as pd
import pprint
import csv
import matplotlib.pyplot as plt
# Define classes
class Party:
    def __init__(self, name: str):
        self.name = name

class Precinct:
    def __init__(self, id: str, name: str, admin_levels: dict,
                 past_electorate: int, current_electorate: int,
                 past_result: dict = None, current_result: dict = None,
                 past_total_votes: float = None, past_invalid_votes: float = None, past_valid_votes: float = None,
                 current_total_votes: float = None, current_invalid_votes: float = None, current_valid_votes: float = None):
        self.id = id
        self.name = name
        self.admin_levels = admin_levels

        self.past_electorate = past_electorate
        self.current_electorate = current_electorate

        self.past_result = past_result or {}
        self.current_result = current_result or {}
        self.projected_result = {}

        self.past_total_votes = past_total_votes
        self.past_invalid_votes = past_invalid_votes
        self.past_valid_votes = past_valid_votes

        self.current_total_votes = current_total_votes
        self.current_invalid_votes = current_invalid_votes
        self.current_valid_votes = current_valid_votes

        self.is_reported = (
            current_result is not None and 
            any(current_result.values()) and 
            current_total_votes not in [None, 0]
        )

class AdminDivision:
    def __init__(self, name: str, level: int, level_name: str):
        self.name = name
        self.level = level
        self.level_name = level_name
        self.precincts = []
        is_lowest_adm_level = False
    
    def add_precinct(self, precinct: Precinct):
        self.precincts.append(precinct)
    
class ElectionMap:
    def __init__(self):
        self.precincts = {}
        self.admin_divisions = {}

    def add_precinct(self, precinct: Precinct):
        self.precincts[precinct.id] = precinct

    def build_admin_hierarchy(self):
        import re
        self.admin_divisions = {}
        lowest_level = -1
        for precinct in self.precincts.values():
            for level_key, division_name in precinct.admin_levels.items():
                match = re.match(r"Adm(\d+)", level_key)
                if not match:
                    continue
                level = int(match.group(1))
                lowest_level = max(lowest_level, level)
                if level not in self.admin_divisions:
                    self.admin_divisions[level] = {}
                if division_name not in self.admin_divisions[level]:
                    self.admin_divisions[level][division_name] = AdminDivision(
                        name=division_name,
                        level=level,
                        level_name=level_key
                    )
                self.admin_divisions[level][division_name].add_precinct(precinct)
        for level, divisions in self.admin_divisions.items():
            for division in divisions.values():
                division.is_lowest_adm_level = (level == lowest_level)

                if level > 0:
                    sample_precinct = division.precincts[0]
                    division.admin_levels_dict = {
                        k: v for k, v in sample_precinct.admin_levels.items() if int(k[3:]) < level
                    }
                else:
                    division.admin_levels_dict = {}

    def get_all_precincts(self):
        return list(self.precincts.values())

    def get_divisions_at_level(self, level: int):
        return self.admin_divisions.get(level, {})

# 1. Load input files
past_df = pd.read_csv("Live projection/previous.csv", sep=";", on_bad_lines="skip")
current_df = pd.read_csv("Live projection/current.csv", sep=";", on_bad_lines="skip")
# Get column lists
adm_columns = [col for col in past_df.columns if col.startswith("Adm")]
base_columns = [
    "precinct_id", "precinct_name", "new_electorate", "past_electorate",
    "total_votes", "invalid_votes", "valid_votes"
]
meta_columns = base_columns + adm_columns
party_columns = [col for col in past_df.columns if col not in meta_columns]
missing_parties = [p for p in party_columns if p not in current_df.columns]
if missing_parties:
    print("Warning: These parties are missing in current results:", missing_parties)
current_df_indexed = current_df.set_index("precinct_id")

# Setting up party objects
party_objects = {name: Party(name) for name in party_columns}

# Create precinct objects
precinct_objects = {}
for _, row in past_df.iterrows():
    precinct_id = row["precinct_id"]
    precinct_name = row["precinct_name"]
    admin_levels = {adm: row[adm] for adm in adm_columns}
    past_electorate = row["past_electorate"]
    new_electorate = row["new_electorate"]
    current_electorate = new_electorate if not pd.isna(new_electorate) else past_electorate
    past_result = {party: row[party] for party in party_columns if not pd.isna(row[party])}
    parties_to_include = [p for p in party_columns if p in current_df.columns]
    if precinct_id in current_df_indexed.index:
        curr_row = current_df_indexed.loc[precinct_id]
        current_result = {}
        for party in parties_to_include:
            value = curr_row.get(party)
            if pd.notna(value):
                current_result[party] = float(value)

        current_total_votes = float(curr_row["total_votes"]) if pd.notna(curr_row.get("total_votes")) else None
        current_invalid_votes = float(curr_row["invalid_votes"]) if pd.notna(curr_row.get("invalid_votes")) else None
        current_valid_votes = float(curr_row["valid_votes"]) if pd.notna(curr_row.get("valid_votes")) else None
    else:
        current_result = None
        current_total_votes = current_invalid_votes = current_valid_votes = None
    
    past_electorate = row["past_electorate"]
    new_electorate = row["new_electorate"]
    current_electorate = new_electorate if not pd.isna(new_electorate) else past_electorate

    precinct_objects[precinct_id] = Precinct(
        id=precinct_id,
        name=precinct_name,
        admin_levels=admin_levels,
        past_electorate=past_electorate,
        current_electorate=current_electorate,
        past_result=past_result,
        current_result=current_result,
        past_total_votes=row["total_votes"],
        past_invalid_votes=row["invalid_votes"],
        past_valid_votes=row["valid_votes"],
        current_total_votes=current_total_votes,
        current_invalid_votes=current_invalid_votes,
        current_valid_votes=current_valid_votes
    )
# Build hierarchy
election_map = ElectionMap()
for precinct in precinct_objects.values():
    election_map.add_precinct(precinct)

election_map.build_admin_hierarchy()
hierarchy_summary = {
    "total_precincts": len(election_map.precincts),
    "admin_levels": list(election_map.admin_divisions.keys()),
    "divisions_per_level": {lvl: len(divs) for lvl, divs in election_map.admin_divisions.items()}
}

# 2. Sum up past and current administrative divisions results

def summarize_votes(division, party_names):

    past_all = {party: 0 for party in party_names}
    past_reported = {party: 0 for party in party_names}
    current = {party: 0 for party in party_names}

    past_total_all = past_valid_all = past_invalid_all = 0
    past_total_reported = past_valid_reported = past_invalid_reported = 0
    current_total = current_valid = current_invalid = 0

    past_electorate_all = 0
    past_electorate_reported = 0
    current_electorate_all = 0
    current_electorate_reported = 0

    for p in division.precincts:
        
        current_electorate_all += p.current_electorate or 0
        past_electorate_all += p.past_electorate or 0

       
        past_total_all += p.past_total_votes or 0
        past_valid_all += p.past_valid_votes or 0
        past_invalid_all += p.past_invalid_votes or 0
        for party in party_names:
            past_all[party] += p.past_result.get(party, 0)

        if p.is_reported:
            
            past_total_reported += p.past_total_votes or 0
            past_valid_reported += p.past_valid_votes or 0
            past_invalid_reported += p.past_invalid_votes or 0
            past_electorate_reported += p.past_electorate or 0
            current_electorate_reported += p.current_electorate or 0

            for party in party_names:
                past_reported[party] += p.past_result.get(party, 0)
                current[party] += int(p.current_result.get(party, 0) or 0)

            current_total += int(p.current_total_votes or 0)
            current_valid += int(p.current_valid_votes or 0)
            current_invalid += int(p.current_invalid_votes or 0)

    return {
        "past_all": {
            "total_votes": past_total_all,
            "valid_votes": past_valid_all,
            "invalid_votes": past_invalid_all,
            "electorate": past_electorate_all,
            "party_votes": past_all
        },
        "past_reported": {
            "total_votes": past_total_reported,
            "valid_votes": past_valid_reported,
            "invalid_votes": past_invalid_reported,
            "electorate": past_electorate_reported,
            "party_votes": past_reported
        },
        "current": {
            "total_votes": current_total,
            "valid_votes": current_valid,
            "invalid_votes": current_invalid,
            "electorate": current_electorate_reported,
            "party_votes": current
        },
        "electorate_totals": {
            "current_electorate_all": current_electorate_all
        }
    }
all_summaries = {}
for level, divisions in election_map.admin_divisions.items():
    all_summaries[level] = {}
    for name, division in divisions.items():
        summary = summarize_votes(division, party_columns)
        all_summaries[level][name] = summary

# 3. Calculate percent reported

for level in all_summaries:
    for name in all_summaries[level]:
        summary = all_summaries[level][name]
        reported = summary["current"]["electorate"]
        total = summary["electorate_totals"]["current_electorate_all"]
        summary["percent_reported"] = reported / total if total else 0

# 4. Calculate share of swing to be applied for lowest level administrative division

def adm_str_to_level(adm_str):
    return int(adm_str.replace("Adm", ""))

exponent = 0.05
swing_weights = {}

for level, divisions in election_map.admin_divisions.items():
    for division in divisions.values():
        if not division.is_lowest_adm_level:
            continue

        hierarchy = [(f"Adm{division.level}", division.name)] + sorted(
            division.admin_levels_dict.items(),
            key=lambda x: int(x[0].replace("Adm", "")),
            reverse=True
        )

        remaining = 1.0
        division_weights = {}

        for adm_key, adm_name in hierarchy[:-1]:
            lvl = adm_str_to_level(adm_key)
            percent_reported = all_summaries[lvl][adm_name]["percent_reported"]
            weight = (percent_reported ** exponent) if percent_reported > 0 else 0
            applied_weight = remaining * weight
            division_weights[adm_name] = applied_weight
            remaining -= applied_weight

        adm0_name = hierarchy[-1][1]
        division_weights[adm0_name] = remaining

        swing_weights[division.name] = division_weights


# 5. Calculate composite vote shares

composite_vote_shares = {}
for level, divisions in election_map.admin_divisions.items():
    for division in divisions.values():
        if not division.is_lowest_adm_level:
            continue

        div_name = division.name
        composite_vote_shares[div_name] = {}
        
        for result_type in ["past_reported", "current"]:
            weighted_turnout = 0
            weighted_valid_share = 0
            weighted_party_shares = {party: 0 for party in party_columns}

            for parent_name, weight in swing_weights.get(div_name, {}).items():
                parent_level = None
                for lvl, divs in election_map.admin_divisions.items():
                    if parent_name in divs:
                        parent_level = lvl
                        break
                if parent_level is None:
                    continue

                parent_summary = all_summaries[parent_level][parent_name].get(result_type, {})
                parent_electorate = all_summaries[parent_level][parent_name][result_type]["electorate"]
                total_votes = parent_summary.get("total_votes", 0)
                valid_votes = parent_summary.get("valid_votes", 0)
                party_votes = parent_summary.get("party_votes", {})

                turnout = total_votes / parent_electorate if parent_electorate else 0
                valid_share = valid_votes / total_votes if total_votes else 0

                weighted_turnout += weight * turnout
                weighted_valid_share += weight * valid_share

                for party in party_columns:
                    share = party_votes.get(party, 0) / valid_votes if valid_votes else 0
                    weighted_party_shares[party] += weight * share

            composite_vote_shares[div_name][result_type] = {
                "turnout": weighted_turnout,
                "valid_share": weighted_valid_share,
                "party_shares": weighted_party_shares
            }

# 6. Calculate composite swing

def calculate_composite_swings(composite_vote_shares, party_columns):
    composite_swings = {}

    for div_name, results in composite_vote_shares.items():
        past = results.get("past_reported", {})
        current = results.get("current", {})

        # Calculate combined changes
        turnout_change = current.get("turnout", 0) - past.get("turnout", 0)
        valid_share_change = current.get("valid_share", 0) - past.get("valid_share", 0)
        party_changes = {}
        for party in party_columns:
            past_share = past.get("party_shares", {}).get(party, 0)
            current_share = current.get("party_shares", {}).get(party, 0)
            party_changes[party] = current_share - past_share

        combined_changes = {
            "turnout": turnout_change,
            "valid_share": valid_share_change,
            "party_shares": party_changes
        }

        # Calculate share kept
        share_kept = {}
        for party in party_columns:
            past_share = past.get("party_shares", {}).get(party, 0)
            current_share = current.get("party_shares", {}).get(party, 0)
            if party_changes[party] < 0 and past_share > 0:
                share_kept[party] = max(min(current_share / past_share, 1.0), 0.0)
            else:
                share_kept[party] = 1.0

        # Calculate gained shares
        gained_shares = {party: max(change, 0) for party, change in party_changes.items()}

        # Calculate share of gains
        total_gains = sum(gained_shares.values())
        if total_gains > 0:
            share_of_gains = {party: gain / total_gains for party, gain in gained_shares.items()}
        else:
            share_of_gains = {party: 0.0 for party in party_columns}

        composite_swings[div_name] = {
            "combined_changes": combined_changes,
            "share_kept": share_kept,
            "gained_shares": gained_shares,
            "share_of_gains": share_of_gains
        }

    return composite_swings
composite_swings = calculate_composite_swings(composite_vote_shares, party_columns)
# 7. Calculate projected result for each precinct

def project_precinct_results(precincts, composite_swings, party_columns):
    lowest_adm_level = max(election_map.admin_divisions.keys())
    for precinct in precincts.values():
        if precinct.is_reported:
            # If precinct already is reported, no projection needs to be made
            precinct.projected_result = {
                "electorate": int(precinct.current_electorate),
                "total_votes": int(precinct.current_total_votes or 0),
                "valid_votes": int(precinct.current_valid_votes or 0),
                "invalid_votes": int(precinct.current_invalid_votes or 0),
                "party_votes": {party: int(precinct.current_result.get(party, 0)) for party in party_columns}
            }
            continue

        lowest_div_name = precinct.admin_levels[f"Adm{lowest_adm_level}"]
        swing_data = composite_swings.get(lowest_div_name, {})
        combined_changes = swing_data.get("combined_changes", {})
        share_kept = swing_data.get("share_kept", {})
        share_of_gains = swing_data.get("share_of_gains", {})

        past_valid = precinct.past_valid_votes or 1
        past_total = precinct.past_total_votes or 1
        past_electorate = precinct.past_electorate or 1
        current_electorate = precinct.current_electorate or past_electorate

        # Determining what share of the vote is kept
        share_kept_in_precinct = {}
        for party in party_columns:
            past_party_votes = precinct.past_result.get(party, 0)
            share_in_precinct = past_party_votes / past_valid if past_valid else 0
            share_kept_in_precinct[party] = share_in_precinct * share_kept.get(party, 0)

        shares_lost = max(0, 1.0 - sum(share_kept_in_precinct.values()))

        # Determining total votes cast and valid votes cast
        turnout_delta = combined_changes.get("turnout", 0)
        projected_turnout = past_total / past_electorate + turnout_delta
        unrounded_total_votes = max(0, current_electorate * projected_turnout)

        valid_share_delta = combined_changes.get("valid_share", 0)
        past_valid_share = (precinct.past_valid_votes or 0) / past_total
        valid_vote_share = past_valid_share + valid_share_delta
        unrounded_valid_votes = max(0, unrounded_total_votes * valid_vote_share)

        # Determining number of votes for each party
        party_votes = {}
        for party in party_columns:
            kept = share_kept_in_precinct.get(party, 0)
            gained = shares_lost * share_of_gains.get(party, 0)
            total_share = kept + gained
            party_votes[party] = max(0, round(unrounded_valid_votes * total_share))

        # Determining valid votes, total votes, and invalid votes
        projected_valid_votes = max(0, sum(party_votes.values()))
        projected_total_votes = max(projected_valid_votes, round(unrounded_total_votes))
        projected_invalid_votes = max(0, projected_total_votes - projected_valid_votes)

        precinct.projected_result = {
            "electorate": current_electorate,
            "total_votes": projected_total_votes,
            "valid_votes": projected_valid_votes,
            "invalid_votes": projected_invalid_votes,
            "party_votes": party_votes
        }
    return precincts
precinct_objects = project_precinct_results(
    precinct_objects,
    composite_swings,
    party_columns,
)

# 8. Sum up projected administrative divisions results


def summarize_projected_votes(division, party_columns):
    projected_total = 0
    projected_valid = 0
    projected_invalid = 0
    projected_electorate = 0
    projected_party_votes = {party: 0 for party in party_columns}

    for p in division.precincts:
        result = p.projected_result
        projected_total += int(result.get("total_votes", 0))
        projected_valid += int(result.get("valid_votes", 0))
        projected_invalid += int(result.get("invalid_votes", 0))
        projected_electorate += int(result.get("electorate", 0))
        for party in party_columns:
            projected_party_votes[party] += int(result.get("party_votes", {}).get(party, 0))

    return {
        "total_votes": projected_total,
        "valid_votes": projected_valid,
        "invalid_votes": projected_invalid,
        "electorate": projected_electorate,
        "party_votes": projected_party_votes
    }
projected_summaries = {}
for level, divisions in election_map.admin_divisions.items():
    projected_summaries[level] = {}
    for name, division in divisions.items():
        projected_summaries[level][name] = summarize_projected_votes(division, party_columns)

# 9. Export as csv

def export_to_csv(
    filename, election_map, projected_summaries, precincts, party_columns
):
    all_levels = sorted(election_map.admin_divisions.keys())
    adm_columns = [f"Adm{lvl}" for lvl in all_levels]

    header = (
        ["id", "name"]
        + adm_columns
        + ["electorate", "total_votes", "invalid_votes", "valid_votes"]
        + party_columns
    )

    rows = []

    for level in all_levels:
        for division_name, division in election_map.admin_divisions[level].items():
            summary = projected_summaries[level][division_name]
            admin_line = {
                "id": f"Adm{level}",
                "name": division.name,
                "electorate": summary["electorate"],
                "total_votes": summary["total_votes"],
                "invalid_votes": summary["invalid_votes"],
                "valid_votes": summary["valid_votes"],
            }

            for lvl in all_levels:
                adm_key = f"Adm{lvl}"
                if lvl == level:
                    admin_line[adm_key] = division.name
                elif lvl < level:
                    admin_line[adm_key] = division.admin_levels_dict.get(adm_key, "")
                else:
                    admin_line[adm_key] = ""

            for party in party_columns:
                admin_line[party] = summary["party_votes"].get(party, 0)

            rows.append(admin_line)

    for precinct in precincts.values():
        result = precinct.projected_result
        precinct_line = {
            "id": precinct.id,
            "name": precinct.name,
            "electorate": result["electorate"],
            "total_votes": result["total_votes"],
            "invalid_votes": result["invalid_votes"],
            "valid_votes": result["valid_votes"],
        }

        for adm_key in adm_columns:
            precinct_line[adm_key] = precinct.admin_levels.get(adm_key, "")

        for party in party_columns:
            precinct_line[party] = result["party_votes"].get(party, 0)

        rows.append(precinct_line)

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    return filename

output_file = "Live projection/projected.csv"
export_to_csv(
    output_file, election_map, projected_summaries, precinct_objects, party_columns
)

# Print summaries for debug

# pprint.pprint(meta_columns, party_columns)
# pprint.pprint(current_df_indexed)
# pprint.pprint(hierarchy_summary)
# pprint.pprint(vars(precinct_objects["A11102"]))
# for level in sorted(election_map.admin_divisions.keys()):
#     print(f"\n=== Administrative Level {level} ===")
#     for division_name, division in election_map.admin_divisions[level].items():
#         print(f"→ {division.level_name} '{division.name}' contains {len(division.precincts)} precincts:")
#         for p in division.precincts:
#             print(f"    - Precinct ID: {p.id}, Name: {p.name}")
# pprint.pprint(vars(election_map.admin_divisions[2]["1. ÅRHUS SYD"]))
# pprint.pprint(all_summaries)
# for level in sorted(all_summaries):
#     print(f"\n=== Level {level} ===")
#     for division_name, summary in all_summaries[level].items():
#         percent = summary.get("percent_reported", 0)
#         print(f"{division_name}: {percent:.2%}")
# pprint.pprint(swing_weights)
# pprint.pprint(composite_vote_shares)
# pprint.pprint(composite_swings)
# pprint.pprint(projected_summaries)
def print_and_save_summary(all_summaries, projected_summaries, party_columns, filename="national_summary.txt"): 
    lines = []

    # National summary
    national_name = list(all_summaries[0].keys())[0]
    current_summary = all_summaries[0][national_name]["current"]
    projected_summary = projected_summaries[0][national_name]

    current_total_votes = current_summary["valid_votes"]
    projected_total_votes = projected_summary["valid_votes"]

    lines.append(f"\nCounted: {all_summaries[0][national_name]['percent_reported']:.2%}\n")
    lines.append(f"{'Candidate':<20} {'Current Votes':>15} {'Current %':>10} {'Projected Votes':>18} {'Projected %':>12}")
    lines.append("-" * 75)

    for party in party_columns:
        curr_votes = current_summary["party_votes"].get(party, 0)
        curr_pct = (curr_votes / current_total_votes) * 100 if current_total_votes else 0
        proj_votes = projected_summary["party_votes"].get(party, 0)
        proj_pct = (proj_votes / projected_total_votes) * 100 if projected_total_votes else 0

        lines.append(f"{party:<20} {curr_votes:>15,} {curr_pct:>9.1f}% {proj_votes:>18,} {proj_pct:>11.1f}%")

    # Administrative divisions reported — FIXED indentation
    lines.append("\nAdministrative divisions that have reported:")
    for level in sorted(all_summaries):
        total = len(all_summaries[level])
        reported = sum(
            1 for name, summary in all_summaries[level].items()
            if summary.get("percent_reported", 0) > 0
        )
        pct = round((reported / total) * 100, 1)
        lines.append(f"Adm{level}: {reported} out of {total} divisions ({pct}%) have at least one reported precinct")

    # Print to console
    for line in lines:
        print(line)

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
def print_and_save_summary(all_summaries, projected_summaries, party_columns, filename="Live projection/summary.txt"): 
    lines = []

    # National summary
    national_name = list(all_summaries[0].keys())[0]
    current_summary = all_summaries[0][national_name]["current"]
    projected_summary = projected_summaries[0][national_name]

    current_total_votes = current_summary["valid_votes"]
    projected_total_votes = projected_summary["valid_votes"]

    lines.append(f"\nCounted: {all_summaries[0][national_name]['percent_reported']:.2%}\n")
    lines.append(f"{'Party':<20} {'Current Votes':>15} {'Current %':>10} {'Projected Votes':>18} {'Projected %':>12}")
    lines.append("-" * 75)

    for party in party_columns:
        curr_votes = current_summary["party_votes"].get(party, 0)
        curr_pct = (curr_votes / current_total_votes) * 100 if current_total_votes else 0
        proj_votes = projected_summary["party_votes"].get(party, 0)
        proj_pct = (proj_votes / projected_total_votes) * 100 if projected_total_votes else 0

        lines.append(f"{party:<20} {curr_votes:>15,} {curr_pct:>9.1f}% {proj_votes:>18,} {proj_pct:>11.1f}%")

    # Administrative divisions reported — FIXED indentation
    lines.append("\nAdministrative divisions that have reported:")
    for level in sorted(all_summaries):
        total = len(all_summaries[level])
        reported = sum(
            1 for name, summary in all_summaries[level].items()
            if summary.get("percent_reported", 0) > 0
        )
        pct = round((reported / total) * 100, 1)
        lines.append(f"Adm{level}: {reported} out of {total} divisions ({pct}%) have at least one reported precinct")

    # Print to console
    for line in lines:
        print(line)

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
print_and_save_summary(all_summaries, projected_summaries, party_columns)
end_time = time.time()
duration = end_time - start_time
print(f"Projection completed in {duration:.2f} seconds.")