import flet as ft

def main(page: ft.Page):
    page.title = "HallBooker"
    page.scroll = "auto"
    page.padding = 0
    page.bgcolor = "#F8F9FA"

    # NAVBAR
    navbar = ft.Container(
        padding=15,
        bgcolor="#F8F9FA",
        content=ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(name="celebration", size=32, color="#0A2463"),
                        ft.Text("HallBooker", size=22, weight="bold", color="#0A2463"),
                    ]
                ),
                ft.Icon(name="menu", size=32, color="#0A2463"),
            ],
        ),
    )

    # HERO SECTION
    hero = ft.Container(
        height=520,
        border_radius=ft.border_radius.all(12),
        image=ft.DecorationImage(
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuAIu-RfZKthTO9VCrL3jG8N9mxNrLFlJ7UB0Q2Z8XQNlsJRDTFtt4JoFKDu4Zuecxp6YPJPe5rrlRlFrUn4rQrbj42U_46yz1x1oBFmoc18RT4mHylH7_Sh2Uwv9Gs4L_ARAUJYArGNnMT14dZeEpG6nARnmfSE0IQypHfiuxg2lhkSPoFOSDb_uvaNiORdOpq2OAcfVlVlZOp3BKiyl4OAoV77yz1BQ60TO7SJMncpgRkrT9YBMmgg8N_mNVlIehu2CT1d1nUcvBU",
            fit=ft.ImageFit.COVER,
        ),
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_center,
            end=ft.alignment.bottom_center,
            colors=["#0A246360", "#0A2463AA"],
        ),
        content=ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            controls=[
                ft.Text(
                    "Your Perfect Event Starts Here",
                    size=34,
                    color="white",
                    weight="bold",
                    text_align="center",
                ),
                ft.Text(
                    "Discover and book unique venues for any occasion.",
                    size=16,
                    color="#E5E5E5",
                    text_align="center",
                ),

                # SEARCH BOX
                ft.Container(
                    width=380,
                    padding=20,
                    border_radius=12,
                    bgcolor="white",
                    content=ft.Column(
                        controls=[
                            ft.Dropdown(
                                label="Event Type",
                                options=[
                                    ft.dropdown.Option("Wedding"),
                                    ft.dropdown.Option("Corporate Event"),
                                    ft.dropdown.Option("Birthday Party"),
                                    ft.dropdown.Option("Anniversary"),
                                ],
                            ),
                            ft.TextField(label="Location"),
                            ft.TextField(label="Date", hint_text="YYYY-MM-DD"),
                            ft.ElevatedButton(
                                "Find a Venue",
                                bgcolor="#FFD700",
                                color="#0A2463",
                                height=45,
                            ),
                        ]
                    ),
                ),
            ],
        ),
    )

    # FEATURES SECTION
    def feature(icon, title, desc):
        return ft.Container(
            padding=15,
            bgcolor="white",
            border_radius=12,
            border=ft.border.all(1, "#DDD"),
            content=ft.Column(
                spacing=5,
                controls=[
                    ft.Icon(icon, size=32, color="#FFD700"),
                    ft.Text(title, size=18, weight="bold", color="#0A2463"),
                    ft.Text(desc, size=14, color="#55648A"),
                ],
            ),
        )

    features = ft.Column(
        spacing=15,
        controls=[
            feature("domain", "Vast Selection", "Explore a wide variety of unique and verified venues for any event."),
            feature("sell", "Transparent Pricing", "What you see is what you get. No hidden fees, ever."),
            feature("check_circle", "Seamless Booking", "Book your dream venue in just a few clicks."),
        ],
    )

    # VENUE CARDS
    def venue_card(title, location, price, image):
        return ft.Container(
            width=240,
            bgcolor="white",
            padding=10,
            border_radius=12,
            shadow=ft.BoxShadow(blur_radius=8, color="#00000030"),
            content=ft.Column(
                spacing=10,
                controls=[
                    ft.Container(
                        height=130,
                        border_radius=8,
                        image=ft.DecorationImage(src=image, fit=ft.ImageFit.COVER),
                    ),
                    ft.Text(title, weight="bold", size=16),
                    ft.Text(location, size=13, color="#55648A"),
                    ft.Text(f"From {price}", weight="bold", size=14),
                    ft.ElevatedButton("View Details"),
                ],
            ),
        )

    venue_row = ft.Row(
        scroll="auto",
        spacing=15,
        controls=[
            venue_card(
                "The Grand Ballroom",
                "New York, NY",
                "$2500",
                "https://lh3.googleusercontent.com/aida-public/AB6AXuAEb0_iDnzLD1Zkhxxo7k6CQUurKtWuH2US15jE4tK0EkhbfhEN-NTh0EyjTc-nY7gbXwC0dESi_lo9cEOdqOnqfk69-lEjLktq91tjqfvTv2gyxbgMd6tPcLicrVQNkUCcIRulXz872JcIPBM2iwOyIh16L--McSFpG6FTCWCzJH_QVnpv2yg9Cno5DVcMdr9ZJaCTyj0y0SciYRq2EnRJti_iWFH5F-YVs-BfAOBhLqltUNlsDPDmEkUrOIShlDI_OG-Vh28K2DI"
            ),
            venue_card(
                "The Skyline Loft",
                "Chicago, IL",
                "$1800",
                "https://lh3.googleusercontent.com/aida-public/AB6AXuDZRbdUduuBTzWOH1oeg5XtWcSBt8Bm-F3JZw_twmX-C8bmQ7EvqSC5aeDROnV9c3aP16RsDoCo3eeMvdgAVwFjuIwpKR32mZ781eBmSepOdUqpovm2kw5sZ6hRby5_-jXs-t90PFi4T1HwRSgY9IIZjidVz0IPebicRjwdtwTDAZAzksbHkQm--WL4o2mRQ9DnNlUdvWAJLaTpfO1PKcxhjxPnp2FtAQNyjg9QASIGKs2Jmf5ifZWalw3MPCGPRsD2R3LNxgFa-Bk"
            ),
            venue_card(
                "The Secret Garden",
                "San Francisco, CA",
                "$3200",
                "https://lh3.googleusercontent.com/aida-public/AB6AXuBRYmR4Bc9fwfjz-frRckbSHG3LqfIxZj-OkJ1bpNpc2OPwKHd2qrst8TYyJP3hERmvgv9rePV_8VFQx7RCLiv9YJkLjgsiQAhjHxJaDZiYp3d8h_1zLOQJ4XXLld6oBC47r_3I0lZ5TSNjTHRRxqO1WgzBQ4QHkjp5WRPe650JiR2vlzjB2gvA4sVFQm_KRjsIzrqt1ZQlMnQTRzpxQn1Wd9kILdXtlmRuzSDmIOOJt4xZyzCz0CgT4tN5tzI92kO9kdPGJHOqu5A"
            ),
        ],
    )

    # FOOTER
    footer = ft.Container(
        padding=20,
        bgcolor="#E6EBFF",
        content=ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text("HallBooker", size=20, weight="bold", color="#0A2463"),
                ft.Text("Contact   |   Terms   |   Privacy", size=14, color="#55648A"),
                ft.Text("© 2024 HallBooker. All Rights Reserved.", size=12, color="#55648A"),
            ],
        ),
    )

    # PAGE LAYOUT
    page.add(
        navbar,
        ft.Container(padding=10, content=hero),
        ft.Container(padding=15, content=features),
        ft.Container(padding=15, content=venue_row),
        footer,
    )

ft.app(target=main)
