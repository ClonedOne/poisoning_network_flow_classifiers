"""
Collection of constants used by all modules.
"""


# Modify the following paths to point to the location where the datasets are stored
base_path = ""
ctu13_base_pth = f"{base_path}/ctu-13"
cicids_base_pth = f"{base_path}/cic-ids-2018"
iscx_base_pth = {
    "file_video": f"{base_path}/ISCXVPN/NonVPN/file_video/",
    "chat_video": f"{base_path}/ISCXVPN/NonVPN/chat_video/",
}

ctu13_res_pth = f"{ctu13_base_pth}/supervised/results"
ctu13_psn_pth = f"{ctu13_base_pth}/supervised/poisoning"
ctu13_psnmm_pth = f"{ctu13_base_pth}/supervised/poisoning_mismatch"
ctu13_evd_pth = f"{ctu13_base_pth}/supervised/evasion"
cicids_res_pth = f"{cicids_base_pth}/supervised/results"
cicids_psn_pth = f"{cicids_base_pth}/supervised/poisoning"
ctu13_conn_log_pth = ctu13_base_pth + "/{}/conn_log.csv"
aggr_feat_pth = ctu13_base_pth + "/supervised/data/{}/neris/window_30/features_stat.csv"


neris_tag = "neris"
iscx_tag = "iscx"
cicids_botnet_tag = "cicids_botnet"

file_names = {
    neris_tag: {
        "AutoEncoder": [
            "neris__2023-02-23_22h53m35s_AutoEncoder_train-1_42-2_43.pkl",
            "neris__2023-02-23_22h53m35s_AutoEncoder_train-1_42-9_50.pkl",
            "neris__2023-02-23_22h53m35s_AutoEncoder_train-2_43-9_50.pkl",
        ],
    }
}
scenario_monikers = {
    neris_tag: [
        "1_42-2_43",
        "1_42-9_50",
        "2_43-9_50",
    ],
}

subscenarios = {
    neris_tag: {
        "train": {
            0: ["1_42", "2_43"],
            1: ["1_42", "9_50"],
            2: ["2_43", "9_50"],
        },
        "test": {
            0: ["9_50"],
            1: ["2_43"],
            2: ["1_42"],
        },
    },
    cicids_botnet_tag: {
        "train": ["friday_02-03-2018_morning"],
        "test": ["friday_02-03-2018_afternoon"],
    },
    iscx_tag: {
        "train": {
            "file_video": ["file_train", "video_train"],
            "chat_video": ["chat_train", "video_train"],
        },
        "test": {
            "file_video": ["file_test", "video_test"],
            "chat_video": ["chat_test", "video_test"],
        },
    },
}

internal_prefix = {
    neris_tag: ("147.32.",),
    cicids_botnet_tag: ("172.31.", "18.219.211.138"),
    iscx_tag: {
        "file_video": ("131.202",),
        "chat_video": ("131.202",),
    },
}
