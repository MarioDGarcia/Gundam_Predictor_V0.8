import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, END, LEFT, RIGHT, X


MODEL_PATH = "modelo_gundam_v3.pkl"

GRADE_CONFIG = {
    "SD": {"escala": "SD", "p_min": 700, "p_max": 1600},
    "HG": {"escala": "1/144", "p_min": 2200, "p_max": 3900},
    "RG": {"escala": "1/144", "p_min": 3200, "p_max": 5200},
    "MG": {"escala": "1/100", "p_min": 4800, "p_max": 9800},
    "PG": {"escala": "1/60", "p_min": 22000, "p_max": 32000},
}

UNIVERSE_MAP = {
    "Universal Century": {
        "subseries": {
            "0079": {"years_since_media": 14, "anniversary_boost": 1},
            "Zeta": {"years_since_media": 22, "anniversary_boost": 0},
            "CCA": {"years_since_media": 18, "anniversary_boost": 0},
            "Unicorn": {"years_since_media": 7, "anniversary_boost": 0},
            "Hathaway": {"years_since_media": 3, "anniversary_boost": 0},
        }
    },
    "Gundam SEED": {
        "subseries": {
            "SEED": {"years_since_media": 3, "anniversary_boost": 0},
            "SEED Destiny": {"years_since_media": 8, "anniversary_boost": 0},
        }
    },
    "Gundam 00": {
        "subseries": {
            "Gundam 00 S1": {"years_since_media": 16, "anniversary_boost": 0},
            "A Wakening of the Trailblazer": {"years_since_media": 14, "anniversary_boost": 0},
        }
    },
    "IBO": {
        "subseries": {
            "Iron-Blooded Orphans": {"years_since_media": 10, "anniversary_boost": 0},
        }
    },
    "Witch from Mercury": {
        "subseries": {
            "WFM S1": {"years_since_media": 2, "anniversary_boost": 0},
            "WFM S2": {"years_since_media": 1, "anniversary_boost": 0},
        }
    },
}

SUIT_ROLES = ["protagonista", "rival", "elite", "grunt", "support"]
RELEASE_TYPES = ["nuevo", "ver_ka", "2.0", "revive", "recolor"]
DISTRIBUTION_TYPES = ["Regular", "P-Bandai", "Event"]

DEFAULT_ROLE_HYPE = {
    "protagonista": 0.72,
    "rival": 0.62,
    "elite": 0.50,
    "grunt": 0.34,
    "support": 0.40,
}

DEFAULT_ROLE_DEMAND = {
    "protagonista": 80,
    "rival": 72,
    "elite": 61,
    "grunt": 44,
    "support": 52,
}

DEFAULT_RELEASE_SATURATION = {
    "nuevo": 0.24,
    "ver_ka": 0.36,
    "2.0": 0.42,
    "revive": 0.49,
    "recolor": 0.76,
}


def get_resource_path(relative_path):
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


modelo = joblib.load(get_resource_path(MODEL_PATH))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def get_price_relative(grado, precio):
    conf = GRADE_CONFIG[grado]
    price_mean = (conf["p_min"] + conf["p_max"]) / 2
    return precio / price_mean


def get_recommendation(score):
    if score >= 0.78:
        return "Muy fuerte dentro de su canal. Parece un lanzamiento muy bien calibrado para su exclusividad."
    if score >= 0.60:
        return "Bueno dentro de su canal. Tiene buen encaje si el precio y la distribucion se mantienen."
    if score >= 0.42:
        return "Medio. Puede funcionar, pero depende mas de timing, suit y saturacion."
    return "Debil para su propio canal. Se parece a un lanzamiento con riesgo de quedarse corto incluso con exclusividad."


def build_input_dataframe():
    grado = combo_grado.get()
    universo = combo_universo.get()
    subserie = combo_subserie.get()
    suit_role = combo_role.get()
    release_type = combo_release.get()
    distribucion = combo_distribution.get()

    precio = float(entry_precio.get())
    anime_on_air = int(var_on_air.get())
    years_since_media = int(scale_years.get())
    anniversary_boost = int(var_anniversary.get())
    saturation = float(scale_saturation.get())
    hype_score = float(scale_hype.get())
    demand_proxy = float(scale_demand.get())

    precio_relativo = get_price_relative(grado, precio)

    return pd.DataFrame(
        [
            {
                "grado": grado,
                "universo": universo,
                "subserie": subserie,
                "suit_role": suit_role,
                "release_type": release_type,
                "distribucion": distribucion,
                "anime_on_air": anime_on_air,
                "years_since_media": years_since_media,
                "anniversary_boost": anniversary_boost,
                "saturation": saturation,
                "precio_relativo": precio_relativo,
                "hype_score": hype_score,
                "demand_proxy": demand_proxy,
            }
        ]
    )


def update_subseries(*_args):
    universo = combo_universo.get()
    subseries = list(UNIVERSE_MAP[universo]["subseries"].keys())
    combo_subserie.configure(values=subseries)
    combo_subserie.set(subseries[0])
    apply_subseries_defaults()


def apply_grade_defaults(*_args):
    grado = combo_grado.get()
    conf = GRADE_CONFIG[grado]
    suggested_price = int((conf["p_min"] + conf["p_max"]) / 2)
    entry_precio.delete(0, END)
    entry_precio.insert(0, str(suggested_price))
    lbl_price_hint.config(text=f"Rango sugerido: ${conf['p_min']:,} - ${conf['p_max']:,}".replace(",", ","))


def apply_subseries_defaults(*_args):
    universo = combo_universo.get()
    subserie = combo_subserie.get()
    defaults = UNIVERSE_MAP[universo]["subseries"][subserie]
    scale_years.set(defaults["years_since_media"])
    var_anniversary.set(defaults["anniversary_boost"])
    lbl_years_value.config(text=str(int(scale_years.get())))


def apply_role_defaults(*_args):
    role = combo_role.get()
    scale_hype.set(DEFAULT_ROLE_HYPE[role])
    scale_demand.set(DEFAULT_ROLE_DEMAND[role])
    lbl_hype_value.config(text=f"{scale_hype.get():.2f}")
    lbl_demand_value.config(text=f"{scale_demand.get():.0f}")


def apply_release_defaults(*_args):
    release_type = combo_release.get()
    scale_saturation.set(DEFAULT_RELEASE_SATURATION[release_type])
    lbl_saturation_value.config(text=f"{scale_saturation.get():.2f}")


def sync_slider_labels(*_args):
    lbl_hype_value.config(text=f"{scale_hype.get():.2f}")
    lbl_saturation_value.config(text=f"{scale_saturation.get():.2f}")
    lbl_demand_value.config(text=f"{scale_demand.get():.0f}")
    lbl_years_value.config(text=f"{int(scale_years.get())}")


def calcular():
    try:
        nuevo = build_input_dataframe()
        pred = float(np.clip(modelo.predict(nuevo)[0], 0, 1))
    except ValueError:
        messagebox.showerror("Dato invalido", "Revisa el precio. Debe ser un numero valido.")
        return
    except Exception as exc:
        messagebox.showerror("Error al predecir", str(exc))
        return

    gauge.configure(amountused=round(pred * 100, 1))
    lbl_status.config(text=f"Exito en su canal: {pred * 100:.1f}%")
    txt_summary.config(state="normal")
    txt_summary.delete("1.0", END)
    txt_summary.insert(
        END,
        (
            f"Universo: {combo_universo.get()} / {combo_subserie.get()}\n"
            f"Configuracion: {combo_grado.get()} | {combo_role.get()} | {combo_release.get()} | {combo_distribution.get()}\n"
            f"Precio relativo: {get_price_relative(combo_grado.get(), float(entry_precio.get())):.2f}\n"
            f"Lectura: {get_recommendation(pred)}"
        ),
    )
    txt_summary.config(state="disabled")


root = tb.Window(themename="darkly", title="Gunpla Market Analyzer v3")
root.geometry("920x760")

main_container = tb.Frame(root)
main_container.pack(fill=BOTH, expand=True)

canvas = tk.Canvas(main_container, highlightthickness=0, bg="#22252b")
scrollbar = tb.Scrollbar(main_container, orient="vertical", command=canvas.yview)
scrollable_frame = tb.Frame(canvas)


def _on_frame_configure(_event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def _on_canvas_configure(event):
    canvas.itemconfigure(canvas_window, width=event.width)


def _on_mousewheel(event):
    delta = int(-1 * (event.delta / 120))
    canvas.yview_scroll(delta, "units")


scrollable_frame.bind("<Configure>", _on_frame_configure)
canvas.bind("<Configure>", _on_canvas_configure)
canvas.bind_all("<MouseWheel>", _on_mousewheel)

canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill="y")

title_frame = tb.Frame(scrollable_frame, padding=(20, 18))
title_frame.pack(fill=X)

tb.Label(title_frame, text="Gunpla Market Analyzer v3", font=("Helvetica", 22, "bold")).pack(anchor="w")
tb.Label(
    title_frame,
    text="Interfaz para el modelo con exito relativo por canal: retail, P-Bandai y event exclusive.",
    font=("Helvetica", 10),
    bootstyle="secondary",
).pack(anchor="w", pady=(4, 0))

body = tb.Frame(scrollable_frame, padding=(20, 8, 20, 20))
body.pack(fill=BOTH, expand=True)

left_panel = tb.Labelframe(body, text="Configuracion del kit", padding=16)
left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

right_panel = tb.Labelframe(body, text="Resultado", padding=16)
right_panel.pack(side=RIGHT, fill=BOTH, expand=True)


def add_labeled_combobox(parent, label_text, values, default_value, command=None):
    tb.Label(parent, text=label_text).pack(anchor="w", pady=(8, 2))
    combo = tb.Combobox(parent, values=values, state="readonly")
    combo.set(default_value)
    combo.pack(fill=X)
    if command is not None:
        combo.bind("<<ComboboxSelected>>", command)
    return combo


def add_scale_row(parent, label_text, from_, to, value, resolution, command=None):
    row = tb.Frame(parent)
    row.pack(fill=X, pady=(8, 0))
    tb.Label(row, text=label_text).pack(anchor="w")
    scale_frame = tb.Frame(parent)
    scale_frame.pack(fill=X)
    scale = tb.Scale(scale_frame, from_=from_, to=to, orient=tk.HORIZONTAL)
    scale.configure(value=value)
    if resolution is not None:
        scale.configure(length=260)
    scale.pack(side=LEFT, fill=X, expand=True)
    value_label = tb.Label(scale_frame, text=str(value), width=7, anchor="e")
    value_label.pack(side=RIGHT, padx=(10, 0))
    if command is not None:
        scale.configure(command=command)
    return scale, value_label


combo_grado = add_labeled_combobox(left_panel, "Grado", list(GRADE_CONFIG.keys()), "HG", apply_grade_defaults)
combo_universo = add_labeled_combobox(left_panel, "Universo", list(UNIVERSE_MAP.keys()), "Universal Century", update_subseries)
combo_subserie = add_labeled_combobox(left_panel, "Subserie", list(UNIVERSE_MAP["Universal Century"]["subseries"].keys()), "0079", apply_subseries_defaults)
combo_role = add_labeled_combobox(left_panel, "Rol del mobile suit", SUIT_ROLES, "protagonista", apply_role_defaults)
combo_release = add_labeled_combobox(left_panel, "Tipo de lanzamiento", RELEASE_TYPES, "nuevo", apply_release_defaults)
combo_distribution = add_labeled_combobox(left_panel, "Distribucion", DISTRIBUTION_TYPES, "Regular")

tb.Label(left_panel, text="Precio").pack(anchor="w", pady=(8, 2))
entry_precio = tb.Entry(left_panel)
entry_precio.pack(fill=X)
lbl_price_hint = tb.Label(left_panel, text="", bootstyle="secondary")
lbl_price_hint.pack(anchor="w", pady=(3, 0))

var_on_air = tb.BooleanVar(value=False)
tb.Checkbutton(
    left_panel,
    text="Anime/serie en emision",
    variable=var_on_air,
    bootstyle="round-toggle",
).pack(anchor="w", pady=(12, 2))

var_anniversary = tb.BooleanVar(value=False)
tb.Checkbutton(
    left_panel,
    text="Boost por aniversario",
    variable=var_anniversary,
    bootstyle="round-toggle",
).pack(anchor="w", pady=(6, 0))

scale_years, lbl_years_value = add_scale_row(left_panel, "Anos desde la ultima media", 0, 25, 3, 1, sync_slider_labels)
scale_saturation, lbl_saturation_value = add_scale_row(left_panel, "Saturacion del mercado", 0, 1, 0.24, 0.01, sync_slider_labels)
scale_hype, lbl_hype_value = add_scale_row(left_panel, "Hype score", 0, 1, 0.72, 0.01, sync_slider_labels)
scale_demand, lbl_demand_value = add_scale_row(left_panel, "Demanda proxy", 5, 99, 80, 1, sync_slider_labels)

tb.Button(
    left_panel,
    text="Analizar lanzamiento",
    command=calcular,
    bootstyle="success",
).pack(fill=X, pady=(18, 0))

gauge = tb.Meter(
    right_panel,
    metersize=240,
    padding=10,
    amountused=0,
    metertype="semi",
    subtext="Exito en su canal",
    bootstyle="info",
    interactive=False,
)
gauge.pack(pady=(10, 8))

lbl_status = tb.Label(right_panel, text="Exito en su canal: 0.0%", font=("Helvetica", 14, "bold"))
lbl_status.pack()

txt_summary = tk.Text(right_panel, height=9, wrap="word", bg="#20242a", fg="#f8f9fa", relief="flat")
txt_summary.pack(fill=X, pady=(18, 0))
txt_summary.insert(
    END,
    "Configura el kit y presiona 'Analizar lanzamiento' para ver la lectura comercial relativa al canal elegido."
)
txt_summary.config(state="disabled")

tips = tb.Labelframe(right_panel, text="Guia rapida", padding=12)
tips.pack(fill=X, pady=(18, 0))
tb.Label(
    tips,
    text=(
        "- Elige el grado, universo y subserie del kit.\n"
        "- Selecciona el rol del mobile suit y el tipo de lanzamiento.\n"
        "- Ajusta hype si el kit viene de algo reciente o muy esperado.\n"
        "- Sube saturacion si ya hay muchas variantes o recolors.\n"
        "- Sube demanda si el suit es muy popular entre fans.\n"
        "- El resultado muestra que tan bien podria funcionar dentro de su propio canal de venta."
    ),
    wraplength=320,
    justify=LEFT,
    bootstyle="secondary",
).pack(anchor="w")

apply_grade_defaults()
update_subseries()
apply_role_defaults()
apply_release_defaults()
sync_slider_labels()
calcular()

root.mainloop()
