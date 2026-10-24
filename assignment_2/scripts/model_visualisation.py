import argparse

import utils.model_visualisation

# to call this script: python scripts/model_visualisation.py --snapshotdate "2025-07-01" --modelname "credit_model_2024_06_01.pkl"


def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')

    # draw monitoring charts with all results known up to snapshotdate (no Spark needed: small table)
    utils.model_visualisation.plot_monitoring_charts(
        snapshotdate,
        model_name=modelname,
        gold_monitoring_directory="datamart/gold/model_monitoring/",
        charts_directory="datamart/gold/model_monitoring_charts/",
    )

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model artefact file name in model_bank/")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
