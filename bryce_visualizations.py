import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def age_vs_month_normalized_elo_diff_scatter_plot(player_data, min_elo_threshold = 2000, min_elo=1400, max_elo=2900):
    ages = []
    normalized_elo_diffs = []

    for fideid, player_months in player_data.items():
        personal_info = preprocessing.get_player_personal_information(player_months)
        sorted_year_months = list(player_months.keys())
        sorted_year_months.sort()

        for i in range(1, len(sorted_year_months)):
            prev_month = player_months[sorted_year_months[i-1]]
            cur_month = player_months[sorted_year_months[i]]

            if prev_month["rating"] is not None and cur_month["rating"] is not None:
                if personal_info.birthday is None:
                    continue

                age = personal_info.get_age_at_datetime(sorted_year_months[i].year_month_to_datetime())
                elo_diff = cur_month["rating"] - prev_month["rating"]
                normalized_elo_diff = elo_diff / (sorted_year_months[i].year_month_to_datetime() - sorted_year_months[i-1].year_month_to_datetime()).days

                if age >= 10 and age <= 70 and normalized_elo_diff <= 1 and normalized_elo_diff >= -1 and normalized_elo_diff != 0:
                    ages.append(age)
                    normalized_elo_diffs.append(normalized_elo_diff)
    
    plt.xlabel("Age")
    plt.ylabel("Normalized Elo Difference")
    plt.title(f"Age vs Normalized Elo Difference Scatter Plot (Players with Max Elo >= {min_elo_threshold})")
    plt.ylim(1400, 2900)
    plt.hist2d(ages, normalized_elo_diffs, bins=30, cmap='Blues')
    plt.savefig(f"visualizations/age_vs_normalized_elo_diff_scatter_plot_{min_elo_threshold}.png")
    plt.show()

def age_vs_elo_heatmap(player_data, elo_threshold = 2000):
    ages = []
    elos = []

    for fideid, player_months in player_data.items():
        personal_info = preprocessing.get_player_personal_information(player_months)
        sorted_year_months = list(player_months.keys())
        sorted_year_months.sort()

        for i in range(1, len(sorted_year_months)):
            prev_month = player_months[sorted_year_months[i-1]]
            cur_month = player_months[sorted_year_months[i]]

            if prev_month["rating"] is not None and cur_month["rating"] is not None:
                if personal_info.birthday is None:
                    continue

                age = personal_info.get_age_at_datetime(sorted_year_months[i].year_month_to_datetime())
                elo = cur_month["rating"]

                if age >= 10 and age <= 70 and elo >= 1500:
                    ages.append(age)
                    elos.append(elo)
    
    plt.xlabel("Age")
    plt.ylabel("Elo")
    plt.title(f"Age vs Elo Heatmap (Players with Max Elo >= {elo_threshold})")
    plt.ylim(1400, 2900)
    plt.hist2d(ages, elos, bins=30, cmap='Blues')
    plt.savefig(f"visualizations/age_vs_elo_heatmap_{elo_threshold}.png")
    plt.show()

def games_played_vs_age(player_data, elo_threshold):
    ages = []
    games_played = []
    activity_ages = []
    any_activity = []

    for fideid, player_months in player_data.items():
        personal_info = preprocessing.get_player_personal_information(player_months)
        sorted_year_months = list(player_months.keys())
        sorted_year_months.sort()

        for i in range(1, len(sorted_year_months)):
            prev_month = player_months[sorted_year_months[i-1]]
            cur_month = player_months[sorted_year_months[i]]

            if prev_month["rating"] is not None and cur_month["rating"] is not None:
                if personal_info.birthday is None:
                    continue

                age = personal_info.get_age_at_datetime(sorted_year_months[i].year_month_to_datetime())
                games = cur_month["games"]

                if age >= 10 and age <= 70:
                    if games is not None and games < 50 and games != 0:
                        ages.append(age)
                        games_played.append(games)
                    
                    activity_ages.append(age)
                    if games is not None and games != 0:
                        any_activity.append(1)
                    else:
                        any_activity.append(0)

    
    plt.xlabel("Age")
    plt.ylabel("Games Played")
    plt.title(f"Games Played vs Age (Players with Max Elo >= {elo_threshold})")
    plt.ylim(1400, 2900)
    plt.hist2d(ages, games_played, bins=30, cmap='Blues')
    plt.savefig(f"visualizations/games_played_vs_age_{elo_threshold}.png")
    plt.show()

    plt.xlabel("Age")
    plt.ylabel("Any Games Played")
    plt.title(f"Any Games Played vs Age (Players with Max Elo >= {elo_threshold})")
    plt.ylim(1400, 2900)
    plt.hist2d(activity_ages, any_activity, bins=30, cmap='Blues')
    plt.savefig(f"visualizations/any_games_played_vs_age_{elo_threshold}.png")
    plt.show()

if __name__ == "__main__":
    for elo_threshold in [2300, 2400, 2500, 2600, 2700]:
        player_data = preprocessing.open_filtered_standard_data(elo_threshold=elo_threshold)
        games_played_vs_age(player_data, elo_threshold=elo_threshold)
        age_vs_elo_heatmap(player_data, elo_threshold=elo_threshold)
        age_vs_month_normalized_elo_diff_scatter_plot(player_data, min_elo_threshold=elo_threshold)