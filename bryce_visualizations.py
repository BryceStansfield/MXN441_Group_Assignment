import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def age_vs_month_normalized_elo_diff_scatter_plot(player_data):
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

                ages.append(age)
                normalized_elo_diffs.append(normalized_elo_diff)
    
    plt.xlabel("Age")
    plt.ylabel("Normalized Elo Difference")
    plt.title("Age vs Normalized Elo Difference Scatter Plot")
    plt.hist2d(ages, normalized_elo_diffs, bins=100, cmap='Blues')
    #sns.jointplot(x=ages, y=normalized_elo_diffs, kind="kde", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    player_data = preprocessing.open_filtered_standard_data(elo_threshold=2300) # I would like a very broad sample for these charts.
    age_vs_month_normalized_elo_diff_scatter_plot(player_data)