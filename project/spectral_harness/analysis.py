# import pandas as pd

# ADAM_PATH = "logs/grid_task_a_full.csv"
# SPEC_PATH = "grid_task_a_full.csv"

# KEYS = ["regime", "seed", "lr", "wd", "eps"]

# adam = pd.read_csv(ADAM_PATH)
# spec = pd.read_csv(SPEC_PATH)

# df = pd.concat([adam, spec], ignore_index=True)

# # Keep final logged row for each config
# final = (
#     df.sort_values("step")
#       .groupby(["regime", "optimizer", "seed", "lr", "wd", "eps"], as_index=False)
#       .tail(1)
# )

# print("\nFinal-loss summary by optimizer:")
# print(final.groupby("optimizer")["loss"].describe())

# print("\nBest final loss by optimizer:")
# print(final.groupby("optimizer")["loss"].min())

# # Matched AdamW vs Spectral comparison
# wide = final.pivot_table(
#     index=KEYS,
#     columns="optimizer",
#     values="loss",
#     aggfunc="first",
# ).reset_index()

# wide = wide.dropna(subset=["adamw", "spectral"])
# wide["spectral_minus_adamw"] = wide["spectral"] - wide["adamw"]
# wide["spectral_wins"] = wide["spectral"] < wide["adamw"]

# print("\nMatched comparison:")
# print(wide["spectral_minus_adamw"].describe())

# print("\nSpectral win rate:")
# print(wide["spectral_wins"].mean())

# print("\nTop 10 spectral wins:")
# print(
#     wide.sort_values("spectral_minus_adamw")
#         .head(10)
#         [KEYS + ["adamw", "spectral", "spectral_minus_adamw"]]
# )

# print("\nTop 10 spectral losses:")
# print(
#     wide.sort_values("spectral_minus_adamw", ascending=False)
#         .head(10)
#         [KEYS + ["adamw", "spectral", "spectral_minus_adamw"]]
# )

# # Mean across seeds for each hyperparameter config
# by_config = (
#     wide.groupby(["regime", "lr", "wd", "eps"], as_index=False)
#         .agg(
#             adamw_mean=("adamw", "mean"),
#             spectral_mean=("spectral", "mean"),
#             spectral_win_rate=("spectral_wins", "mean"),
#         )
# )

# by_config["spectral_minus_adamw_mean"] = (
#     by_config["spectral_mean"] - by_config["adamw_mean"]
# )

# print("\nBest configs by mean spectral advantage:")
# print(
#     by_config.sort_values("spectral_minus_adamw_mean")
#         .head(10)
# )

# wide.to_csv("logs/task_b_matched_comparison.csv", index=False)
# by_config.to_csv("logs/task_b_by_config_summary.csv", index=False)

# print("\nWrote:")
# print("logs/task_b_matched_comparison.csv")
# print("logs/task_b_by_config_summary.csv")


# import pandas as pd

# PATH = "grid_task_a_full.csv"

# KEYS = ["regime", "seed", "lr", "wd", "eps"]

# df = pd.read_csv(PATH)

# # Keep final logged row for each run
# final = (
#     df.sort_values("step")
#       .groupby(["regime", "optimizer", "seed", "lr", "wd", "eps"], as_index=False)
#       .tail(1)
# )

# print("\nFinal-loss summary by optimizer:")
# print(final.groupby("optimizer")["loss"].describe())

# print("\nBest final loss by optimizer:")
# print(final.groupby("optimizer")["loss"].min())

# # Matched AdamW vs Spectral comparison
# wide = final.pivot_table(
#     index=KEYS,
#     columns="optimizer",
#     values="loss",
#     aggfunc="first",
# ).reset_index()

# wide = wide.dropna(subset=["adamw", "spectral"])

# wide["spectral_minus_adamw"] = (
#     wide["spectral"] - wide["adamw"]
# )

# wide["spectral_wins"] = (
#     wide["spectral"] < wide["adamw"]
# )

# print("\nMatched comparison:")
# print(wide["spectral_minus_adamw"].describe())

# print("\nSpectral win rate:")
# print(wide["spectral_wins"].mean())

# print("\nTop 10 spectral wins:")
# print(
#     wide.sort_values("spectral_minus_adamw")
#         .head(10)
#         [KEYS + ["adamw", "spectral", "spectral_minus_adamw"]]
# )

# print("\nTop 10 spectral losses:")
# print(
#     wide.sort_values("spectral_minus_adamw", ascending=False)
#         .head(10)
#         [KEYS + ["adamw", "spectral", "spectral_minus_adamw"]]
# )

# # Aggregate across seeds
# by_config = (
#     wide.groupby(["regime", "lr", "wd", "eps"], as_index=False)
#         .agg(
#             adamw_mean=("adamw", "mean"),
#             spectral_mean=("spectral", "mean"),
#             spectral_win_rate=("spectral_wins", "mean"),
#         )
# )

# by_config["spectral_minus_adamw_mean"] = (
#     by_config["spectral_mean"] - by_config["adamw_mean"]
# )

# print("\nBest configs by mean spectral advantage:")
# print(
#     by_config.sort_values("spectral_minus_adamw_mean")
#         .head(10)
# )


import pandas as pd

df1 = pd.read_csv("grid_task_a_full.csv")
df2 = pd.read_csv("logs/grid_task_a_full.csv")

combined = pd.concat([df1, df2], ignore_index=True)

combined.to_csv("grid_task_a_full.csv", index=False)