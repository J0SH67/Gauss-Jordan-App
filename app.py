import numpy as np
import pandas as pd
import streamlit as st
from gauss_jordan import solve_gauss_jordan
from linear_regression import MultipleLinearRegression

# -----------------------------------------------------------------------------
# HELPER: UNICODE SUBSCRIPT GENERATOR (x₁ instead of x_1)
# -----------------------------------------------------------------------------
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Civil Engineering Matrix & ML Solver",
    page_icon="🏗️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean rounded cards for metrics */
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }
    
    /* Interactive hover lift for primary buttons */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.25s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.4);
    }

    /* Top banner styling */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #f59e0b;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 24px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & GROUP MEMBERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Project Info")
    st.markdown("**Department:** Civil Engineering")
    st.markdown("**Core Method:** Gauss-Jordan Elimination")
    st.markdown("**ML Model:** Multiple Linear Regression (OLS)")

    st.divider()

    st.subheader("👥 Group Members")
    st.markdown("• **John Joshua D. Ilisan**")
    st.markdown("• **Darius Lape**")
    st.markdown("• **Briel Jan M. Lacia**")

    st.divider()
    st.caption("Engineered for offline and online structural analysis.")

# -----------------------------------------------------------------------------
# 4. TOP HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h2 style="margin: 0; color: #F8FAFC;">Numerical Solver & ML Engine</h2>
        <p style="margin: 6px 0 10px 0; color: #94A3B8;">Gauss-Jordan Elimination with Partial Pivoting.</p>
        <p style="margin: 0; font-size: 0.9rem; color: #F59E0B;">
            <strong>Proponents:</strong> John Joshua D. Ilisan | Darius Lape | Briel Jan M. Lacia
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(
    ["🔢 System of Linear Equations", "📈 Multiple Linear Regression"]
)

# -----------------------------------------------------------------------------
# TAB 1: GAUSS-JORDAN MATRIX SOLVER
# -----------------------------------------------------------------------------
with tab1:
    st.header("Solve a System of Linear Equations ($Ax = b$)")

    with st.container(border=True):
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            n = st.number_input(
                "Matrix Dimension (n x n):",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
            )
        with col_ctrl2:
            st.info(
                "Click any cell in the table below to edit coefficients. Always"
                " press **Enter** after typing."
            )

        default_data = [
            [4.0, -1.0, 0.0, 2.0, 15.0],
            [-1.0, 5.0, -2.0, 0.0, 10.0],
            [0.0, -2.0, 6.0, -1.0, 8.0],
            [2.0, 0.0, -1.0, 4.0, 20.0],
        ]

        col_headers = [f"x{str(i+1).translate(sub)}" for i in range(n)] + [
            "Constants (b)"
        ]

        if n == 4:
            df_init = pd.DataFrame(default_data, columns=col_headers)
        else:
            df_init = pd.DataFrame(0.0, index=range(n), columns=col_headers)

        edited_df = st.data_editor(
            df_init,
            use_container_width=True,
            num_rows="fixed",
            key=f"matrix_editor_{n}",
        )

        solve_clicked = st.button(
            "Solve with Gauss-Jordan", type="primary", use_container_width=True
        )

    if solve_clicked:
        try:
            matrix_vals = edited_df.values
            A = matrix_vals[:, :n].tolist()
            b = matrix_vals[:, n].tolist()

            solution, steps = solve_gauss_jordan(A, b, return_steps=True)

            st.success("✅ System successfully reduced to identity matrix!")

            cols = st.columns(n)
            for i in range(n):
                with cols[i]:
                    var_name = f"x{str(i+1).translate(sub)}"
                    st.metric(label=var_name, value=f"{solution[i]:.4f}")

            st.divider()
            st.subheader("📝 Step-by-Step Row Operations")
            st.caption(f"Reduced in {len(steps)} systematic transformations.")

            for idx, step_info in enumerate(steps):
                with st.expander(
                    f"Step {idx+1}: {step_info['title']}", expanded=(idx == 0)
                ):
                    st.markdown(f"**Action:** `{step_info['operation']}`")
                    step_df = pd.DataFrame(
                        step_info["matrix"], columns=col_headers
                    )
                    st.dataframe(
                        step_df.style.format("{:.4f}"), use_container_width=True
                    )

        except Exception as e:
            st.error(f"Solver Error: {e}")

# -----------------------------------------------------------------------------
# TAB 2: MULTIPLE LINEAR REGRESSION (ML LAYER)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Concrete Strength Predictor")
    st.caption(
        "Fits Ordinary Least Squares via the Normal Equations using our custom"
        " Gauss-Jordan solver."
    )

    with st.container(border=True):
        st.subheader("1. Training Dataset (Mix Batch Observations)")
        default_ml_data = {
            "Cement_kg_m3": [
                280.0,
                280.0,
                350.0,
                350.0,
                420.0,
                420.0,
                310.0,
                390.0,
            ],
            "Water_Cement_Ratio": [
                0.50,
                0.50,
                0.42,
                0.42,
                0.35,
                0.35,
                0.48,
                0.38,
            ],
            "Curing_Age_Days": [7.0, 28.0, 7.0, 28.0, 7.0, 28.0, 14.0, 14.0],
            "Strength_MPa": [21.5, 32.0, 29.8, 41.2, 38.5, 52.0, 28.4, 43.1],
        }
        data_df = pd.DataFrame(default_ml_data)
        edited_data = st.data_editor(data_df, use_container_width=True)

        train_clicked = st.button(
            "🧠 Train Regression Model",
            type="primary",
            use_container_width=True,
        )

    if train_clicked:
        try:
            feature_cols = [
                "Cement_kg_m3",
                "Water_Cement_Ratio",
                "Curing_Age_Days",
            ]
            X = edited_data[feature_cols].values
            y = edited_data["Strength_MPa"].values

            model = MultipleLinearRegression()
            coeffs, ml_steps = model.fit(X, y, return_steps=True)

            st.session_state["trained_model"] = model
            st.session_state["coeffs"] = coeffs
            st.session_state["ml_steps"] = ml_steps
            st.session_state["XT_X"] = model.XT_X
            st.session_state["XT_y"] = model.XT_y
            st.session_state["X_data"] = X
            st.session_state["y_data"] = y

            st.success("✅ Model weights optimized successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")

    if "trained_model" in st.session_state:
        coeffs = st.session_state["coeffs"]
        ml_steps = st.session_state["ml_steps"]
        XT_X = st.session_state["XT_X"]
        XT_y = st.session_state["XT_y"]
        X_data = st.session_state["X_data"]
        y_data = st.session_state["y_data"]

        with st.container(border=True):
            st.subheader("2. Learned Model Weights")
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("Intercept (β₀)", f"{coeffs[0]:.4f}")
            c1.metric("Cement (β₁)", f"{coeffs[1]:.4f}")
            c2.metric("w/c Ratio (β₂)", f"{coeffs[2]:.4f}")
            c3.metric("Age Days (β₃)", f"{coeffs[3]:.4f}")

            st.latex(
                rf"\text{{Strength}} = {coeffs[0]:.2f} + ({coeffs[1]:.4f} \times \text{{Cement}}) + ({coeffs[2]:.2f} \times \text{{w/c}}) + ({coeffs[3]:.4f} \times \text{{Age}})"
            )

        # ---------------------------------------------------------------------
        # 3. MATHEMATICAL DERIVATION: (XᵀX)β = Xᵀy
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader(
                "3. 📝 Step-by-Step Derivation of $(X^T X)\\beta = X^T y$"
            )
            st.markdown(
                "Training this model minimizes the squared prediction errors"
                " using the **Normal Equations**:"
            )
            st.latex(r"(X^T X)\beta = X^T y \quad \iff \quad A\beta = b")

            x1 = X_data[:, 0]
            x2 = X_data[:, 1]
            x3 = X_data[:, 2]
            y_vec = y_data
            N = len(y_vec)

            with st.expander(
                "📌 Part A: Matrix Equations (Symbolic Formula vs. Numerical"
                " Values)",
                expanded=True,
            ):
                st.markdown("**1. General Symbolic Matrix Structure:**")
                st.latex(r"""
                \begin{bmatrix}
                N & \sum x_1 & \sum x_2 & \sum x_3 \\
                \sum x_1 & \sum x_1^2 & \sum x_1 x_2 & \sum x_1 x_3 \\
                \sum x_2 & \sum x_2 x_1 & \sum x_2^2 & \sum x_2 x_3 \\
                \sum x_3 & \sum x_3 x_1 & \sum x_3 x_2 & \sum x_3^2
                \end{bmatrix}
                \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \beta_3 \end{bmatrix}
                =
                \begin{bmatrix}
                \sum y \\
                \sum x_1 y \\
                \sum x_2 y \\
                \sum x_3 y
                \end{bmatrix}
                """)

                st.markdown(
                    "**2. Substituted with Computed Numerical Values:**"
                )
                st.latex(rf"""
                \begin{{bmatrix}}
                {XT_X[0,0]:.1f} & {XT_X[0,1]:.1f} & {XT_X[0,2]:.4f} & {XT_X[0,3]:.1f} \\
                {XT_X[1,0]:.1f} & {XT_X[1,1]:,.1f} & {XT_X[1,2]:,.1f} & {XT_X[1,3]:,.1f} \\
                {XT_X[2,0]:.4f} & {XT_X[2,1]:,.1f} & {XT_X[2,2]:.4f} & {XT_X[2,3]:.4f} \\
                {XT_X[3,0]:.1f} & {XT_X[3,1]:,.1f} & {XT_X[3,2]:.4f} & {XT_X[3,3]:,.1f}
                \end{{bmatrix}}
                \begin{{bmatrix}} \beta_0 \\ \beta_1 \\ \beta_2 \\ \beta_3 \end{{bmatrix}}
                =
                \begin{{bmatrix}}
                {XT_y[0]:.1f} \\
                {XT_y[1]:,.1f} \\
                {XT_y[2]:.3f} \\
                {XT_y[3]:,.1f}
                \end{{bmatrix}}
                """)

            with st.expander(
                "🔢 Part B: Calculations Breakdown (Data Values &"
                " Arithmetic)",
                expanded=True,
            ):
                st.markdown(
                    "Each entry in the matrix is formed by summing across all"
                    " batch samples:"
                )

                def format_sum(arr):
                    if len(arr) <= 8:
                        return " + ".join([f"{v:g}" for v in arr])
                    return (
                        " + ".join([f"{v:g}" for v in arr[:4]])
                        + " + ... + "
                        + " + ".join([f"{v:g}" for v in arr[-2:]])
                    )

                def format_prod_sum(arr1, arr2):
                    if len(arr1) <= 8:
                        return " + ".join(
                            [f"({a:g}×{b:g})" for a, b in zip(arr1, arr2)]
                        )
                    first = " + ".join(
                        [f"({a:g}×{b:g})" for a, b in zip(arr1[:3], arr2[:3])]
                    )
                    return f"{first} + ... + ({arr1[-1]:g}×{arr2[-1]:g})"

                sum_table = {
                    "Matrix Term": [
                        "N (Sample Count)",
                        "Σ Cement (x₁)",
                        "Σ w/c Ratio (x₂)",
                        "Σ Age (x₃)",
                        "Σ Strength (y)",
                        "Σ Cement² (x₁²)",
                        "Σ (Cement × w/c)",
                        "Σ (Cement × Age)",
                        "Σ (Cement × Strength)",
                        "Σ (w/c)² (x₂²)",
                        "Σ (w/c × Age)",
                        "Σ (w/c × Strength)",
                        "Σ Age² (x₃²)",
                        "Σ (Age × Strength)",
                    ],
                    "Actual Numbers Being Added": [
                        "1 + 1 + 1 + 1 + 1 + 1 + 1 + 1",
                        format_sum(x1),
                        format_sum(x2),
                        format_sum(x3),
                        format_sum(y_vec),
                        format_sum(np.round(x1**2, 1)),
                        format_prod_sum(x1, x2),
                        format_prod_sum(x1, x3),
                        format_prod_sum(x1, y_vec),
                        format_sum(np.round(x2**2, 4)),
                        format_prod_sum(x2, x3),
                        format_prod_sum(x2, y_vec),
                        format_sum(np.round(x3**2, 1)),
                        format_prod_sum(x3, y_vec),
                    ],
                    "Computed Total": [
                        f"{N}",
                        f"{np.sum(x1):,.2f}",
                        f"{np.sum(x2):,.4f}",
                        f"{np.sum(x3):,.1f}",
                        f"{np.sum(y_vec):,.2f}",
                        f"{np.sum(x1**2):,.2f}",
                        f"{np.sum(x1*x2):,.2f}",
                        f"{np.sum(x1*x3):,.2f}",
                        f"{np.sum(x1*y_vec):,.2f}",
                        f"{np.sum(x2**2):,.4f}",
                        f"{np.sum(x2*x3):,.2f}",
                        f"{np.sum(x2*y_vec):,.2f}",
                        f"{np.sum(x3**2):,.2f}",
                        f"{np.sum(x3*y_vec):,.2f}",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(sum_table),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown(
                "##### **Part C: Assembled Augmented Matrix $[(X^T X) \\mid"                 " (X^T y)]$**"
            )
            norm_col_headers = [
                "β₀ (Intercept)",
                "β₁ (Cement)",
                "β₂ (w/c)",
                "β₃ (Age)",
                "Constants (Xᵀy)",
            ]
            initial_augmented = np.hstack([XT_X, XT_y.reshape(-1, 1)])
            df_norm = pd.DataFrame(
                initial_augmented,
                columns=norm_col_headers,
                index=[
                    "Row 1 (β₀)",
                    "Row 2 (β₁)",
                    "Row 3 (β₂)",
                    "Row 4 (β₃)",
                ],
            )
            st.dataframe(
                df_norm.style.format("{:,.4f}"), use_container_width=True
            )

            st.divider()
            st.markdown(
                "##### **Part D: Step-by-Step Gauss-Jordan Reduction to"
                " Identity $[I \\mid \\beta]$**"
            )
            st.caption(
                f"Our custom solver reduces this 4×4 system in {len(ml_steps)}"
                " steps to solve for the final coefficients:"
            )

            for idx, step_info in enumerate(ml_steps):
                with st.expander(
                    f"Step {idx+1}: {step_info['title']}", expanded=(idx == 0)
                ):
                    st.markdown(f"**Action:** `{step_info['operation']}`")
                    step_df = pd.DataFrame(
                        step_info["matrix"],
                        columns=norm_col_headers,
                        index=["Row 1", "Row 2", "Row 3", "Row 4"],
                    )
                    st.dataframe(
                        step_df.style.format("{:,.4f}"),
                        use_container_width=True,
                    )

        # ---------------------------------------------------------------------
        # 4. ENHANCED MODEL VALIDATION (DETAILED ACTUAL VS. PREDICTED)
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader("4. 📊 Model Validation: Actual vs. Predicted Analysis")
            st.markdown(
                "This section evaluates how accurately our trained Gauss-Jordan"
                " regression model predicts actual laboratory cylinder test"
                " breaks."
            )

            # Compute predictions and statistical accuracy metrics
            y_pred = st.session_state["trained_model"].predict(X_data)
            residuals = y_data - y_pred
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            ss_res = np.sum(residuals**2)
            r2 = 1.0 - (ss_res / ss_tot)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))

            # Display Key Statistical Performance Indicators (KPIs)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "R² Score (Fit Quality)",
                f"{r2:.4f}",
                help="1.0 indicates a perfect fit. >0.90 indicates high predictive accuracy.",
            )
            m2.metric(
                "Mean Absolute Error (MAE)",
                f"{mae:.2f} MPa",
                help="Average discrepancy between predicted and actual laboratory breaks.",
            )
            m3.metric(
                "Root Mean Squared Error (RMSE)",
                f"{rmse:.2f} MPa",
                help="Standard deviation of residuals; penalizes larger errors.",
            )
            m4.metric(
                "Max Error",
                f"{np.max(np.abs(residuals)):.2f} MPa",
                help="Largest single variance across all test batches.",
            )

            # Axis Explanation Banner
            st.info("""
            **Understanding the Graph Below:**
            * 📌 **Horizontal Axis (X-Axis):** **Batch Cylinder Specimen** (`Batch 1` to `Batch 8`). Each point corresponds to a distinct laboratory mix proportion.
            * 📌 **Vertical Axis (Y-Axis):** **Compressive Strength in Megapascals ($\text{MPa}$)**.
            * 🔵 **Blue Line:** Actual Lab Measured Strength (from Universal Testing Machine breaks).
            * 🔴 **Red Line:** Model Predicted Strength (computed using the Gauss-Jordan weights).
            """)

            batch_labels = [f"Batch {i+1}" for i in range(len(y_data))]
            chart_df = pd.DataFrame(
                {
                    "Actual Lab Strength (MPa)": y_data,
                    "Model Prediction (MPa)": y_pred,
                },
                index=batch_labels,
            )

            st.line_chart(
                chart_df,
                 color=["#3b82f6", "#ef4444"],  # #3b82f6 = Blue (Actual), #ef4444 = Red (Predicted)
                 use_container_width=True,
)   

            # Detailed Specimen-by-Specimen Audit Table
            with st.expander(
                "📋 View Detailed Batch-by-Batch Residual Error Table",
                expanded=True,
            ):
                audit_df = pd.DataFrame(
                    {
                        "Batch Specimen": batch_labels,
                        "Cement (kg/m³)": X_data[:, 0],
                        "w/c Ratio": X_data[:, 1],
                        "Curing Age (Days)": X_data[:, 2],
                        "Actual Strength (MPa)": y_data,
                        "Predicted Strength (MPa)": y_pred,
                        "Residual Error (MPa)": residuals,
                        "Accuracy (%)": 100.0
                        * (1.0 - (np.abs(residuals) / y_data)),
                    }
                )

                st.dataframe(
                    audit_df.style.format({
                        "Cement (kg/m³)": "{:.1f}",
                        "w/c Ratio": "{:.2f}",
                        "Curing Age (Days)": "{:.0f}",
                        "Actual Strength (MPa)": "{:.2f}",
                        "Predicted Strength (MPa)": "{:.2f}",
                        "Residual Error (MPa)": "{:+.2f}",
                        "Accuracy (%)": "{:.1f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        # ---------------------------------------------------------------------
        # 5. INTERACTIVE MIX DESIGN PREDICTOR
        # ---------------------------------------------------------------------
        with st.container(border=True):
            st.subheader("5. 🎛️ Interactive Strength Predictor")
            st.write(
                "Use the sliders below to test arbitrary mix designs in real"
                " time:"
            )

            p1, p2, p3 = st.columns(3)
            with p1:
                in_cement = st.slider(
                    "Cement Content (kg/m³)", 150.0, 550.0, 350.0, 10.0
                )
            with p2:
                in_wc = st.slider(
                    "Water-Cement Ratio (w/c)", 0.25, 0.65, 0.42, 0.01
                )
            with p3:
                in_age = st.slider("Curing Age (Days)", 1.0, 90.0, 28.0, 1.0)

            sample = [[in_cement, in_wc, in_age]]
            pred_strength = st.session_state["trained_model"].predict(sample)[0]

            st.markdown(
                f"""
                <div style="background-color: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #f59e0b; text-align: center; margin-top: 10px;">
                    <span style="font-size: 1.1rem; color: #94A3B8;">Estimated 28-Day Strength:</span>
                    <h2 style="margin: 4px 0 0 0; color: #F59E0B;">{pred_strength:.2f} MPa</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )