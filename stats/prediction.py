import pickle


rd = lambda x: round(x * 100, 1)

with open(r"./plots/flag_data.pkl", "rb") as f:
    table = pickle.load(f)

# Two tables (for Suppl. Information)
top_str = r"""\clearpage
\begin{table}[ht!]
\centering"""

red_top_str = r"""\begin{tabular}[t]{@{} r | c c c c | c c c c | c c c c @{}}
\multicolumn{13}{l}{\textbf{{\large a}~~Red flag thresholds}} \\"""

yellow_top_str = r"""\begin{tabular}[t]{@{} r | c c c c | c c c c | c c c c @{}}
\multicolumn{13}{l}{\textbf{{\large a}~~Yellow flag thresholds}} \\"""

common_top_str = r"""\toprule
& \multicolumn{4}{c}{\textbf{D-Network}} & \multicolumn{4}{c}{\textbf{R-Network}} & \multicolumn{4}{c}{\textbf{Full}} \\
& \multicolumn{2}{c}{Survivors} & \multicolumn{2}{c}{Nonsurvivors} & \multicolumn{2}{c}{Survivors} & 
\multicolumn{2}{c}{Nonsurvivors} & \multicolumn{2}{c}{Survivors} & \multicolumn{2}{c}{Nonsurvivors} \\
& $Q_{D}$ & $V_{D}$ & $Q_{D}$ & $V_{D}$ & $Q_{R}$ & $V_{R}$ & $Q_{R}$ & $V_{R}$ & $Q$ & $V$ & $Q$ & $V$ \\"""

mid_end_str = r"""\bottomrule
\multicolumn{13}{l}{} \\
\end{tabular}"""

end_str = r"""\bottomrule
\end{tabular}
\caption{Prediction of potentially harmful selected treatments from D-Network with the threshold of $\delta_{D}=-0.3$ ...}
\label{table:prediction}
\end{table}"""

def writer(table, flag):
    for t, time_str in enumerate(table["time"]):
        text = r"\textbf{" + time_str[:-6] + r" h}"
        for net in ["D", "R", "FULL"]:
            for traj_type in ["survivors", "nonsurvivors"]:
                text += r" & "
                text += str(rd(table[traj_type]["Q_"+net][flag][t])) + r"\%"
                text += r" & "
                text += str(rd(table[traj_type]["V_"+net][flag][t])) + r"\%"
        text += r" \\"
        f.write(text)
        f.write("\n")

with open("plots/latex_suppl.txt", "w") as f:
    f.write(top_str)
    f.write("\n")
    f.write(red_top_str)
    f.write(common_top_str)
    f.write("\n")
    writer(table, "red")
    f.write(mid_end_str)
    f.write("\n")
    f.write(yellow_top_str)
    f.write(common_top_str)
    f.write("\n")
    writer(table, "yellow")
    f.write(end_str)

# More concise version of the table (Red+Yellow for Full in one table) -- NOT used in the paper.
top_str = r"""\clearpage
\begin{table}[ht!]
\centering

\begin{tabular}[t]{@{} r | c c c c | c c c c @{}}
\multicolumn{9}{l}{\textbf{Confirmed patients with red and yellow flags.}} \\
\toprule
& \multicolumn{4}{c}{\textbf{Yellow Flag}} & \multicolumn{4}{c}{\textbf{Red Flag}} \\
Time & \multicolumn{2}{c}{Survivors} & 
\multicolumn{2}{c}{Nonsurvivors} & \multicolumn{2}{c}{Survivors} & \multicolumn{2}{c}{Nonsurvivors} \\
& $Q$ & $V$ & $Q$ & $V$ & $Q$ & $V$ & $Q$ & $V$ \\"""

end_str = r"""\bottomrule
\end{tabular}
\caption{\textbf{Prediction of potentially life-threatening treatments and states.} Similarly ...}
\label{table:prediction:main}
\end{table}"""

with open("plots/latex_main.txt", "w") as f:
    f.write(top_str)
    f.write("\n")
    for t, time_str in enumerate(table["time"]):
        text = r"\textbf{" + time_str[:-6] + r" h}"
        for flag in ["yellow", "red"]:
            for traj_type in ["survivors", "nonsurvivors"]:
                text += r" & "
                text += str(rd(table[traj_type]["Q_FULL"][flag][t])) + r"\%"
                text += r" & "
                text += str(rd(table[traj_type]["V_FULL"][flag][t])) + r"\%"
        text += r" \\"
        f.write(text)
        f.write("\n")
    f.write("\n")
    f.write(end_str)