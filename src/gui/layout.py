import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
            .stApp {
                background-image: url('https://res.cloudinary.com/dykxtkzm8/image/upload/v1754630959/WhatsApp_Image_2025-08-08_at_10.49.50_AM_gv5cm2.jpg');
                background-size: cover;
                background-attachment: fixed;
            }
            textarea {
                background-color: #f59e0b !important;
                color: black !important;
                font-weight: 500 !important;
            }
            .stButton > button {
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 24px;
                margin: 10px 10px 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color:  #001f3f;
            color: white;
            border-radius: 10px;
            height: 50px;
            font-size: 18px;
        }
        div.stButton > button:first-child:hover {
            background-color: #004080;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


def show_header():
    st.markdown("## GreenMind: Energy-Efficient Prompt and Context Engineering")
    st.caption("Check the predicted Energy Consumption and hit **Improve** to see a more energy-efficient Prompt.")


def get_prompt_input():
    st.markdown("### Enter prompt here:")
    return st.text_area(
        " ",
        placeholder="""Role -----------------------
    I am... Lorem ipsum dolor sit amet
    You are... Consectetur adipiscing elit

    Context --------------
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

    Expectations-------
    Ut enim ad minim veniam""",
        height=220,
        label_visibility='collapsed'
    )


def get_parameter_inputs():
    st.markdown("### Set parameters:")
    num_layers = st.number_input("# Layers", min_value=1, value=4)
    training_hours = st.number_input("Training time (hrs)", min_value=1, value=2)
    flops_per_hour = st.number_input(
        "FLOPs/hr.",
        min_value=1e5,
        value=1e20,
        step=1e18,
        format="%.2e"
    )
    return num_layers, training_hours, flops_per_hour


def action_buttons():
    col_submit, col_improve = st.columns([1, 1])
    submit_clicked = col_submit.button("Submit")
    improve_clicked = col_improve.button("Improve", use_container_width=True)
    return submit_clicked, improve_clicked