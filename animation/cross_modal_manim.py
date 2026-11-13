from manim import *

class CrossModalArchitecture(Scene):
    def construct(self):
        # 3Blue1Brown signature dark slate background & palette
        self.camera.background_color = "#0e1117"

        C_SAR = "#58C4DD"      # 3b1b Blue / Cyan
        C_OPT = "#F0AC5F"      # 3b1b Gold / Orange
        C_ONLINE = "#83C167"   # 3b1b Green
        C_TARGET = "#5CD0B3"   # 3b1b Teal
        C_GEM = "#9A72AC"      # 3b1b Purple
        C_LOSS = "#FC6255"     # 3b1b Red / Coral
        C_PANEL = "#161b22"    # Dark card fill
        C_BORDER = "#30363d"   # Subtle slate border

        # --- Helper: Layered Tensor Stack (3b1b style) ---
        def make_tensor_stack(num_layers, side_len, color, title_tex, shape_tex):
            layers = VGroup()
            dx, dy = 0.055, 0.045
            for i in range(num_layers):
                sq = Square(
                    side_length=side_len,
                    stroke_color=color,
                    stroke_width=1.8,
                    fill_color=color,
                    fill_opacity=0.18,
                ).shift(RIGHT * i * dx + UP * i * dy)
                layers.add(sq)
            layers.move_to(ORIGIN)
            t_lbl = Tex(title_tex, color=color).scale(0.42).next_to(layers, UP, buff=0.12)
            s_lbl = MathTex(shape_tex, color=GREY_B).scale(0.38).next_to(layers, DOWN, buff=0.12)
            return VGroup(layers, t_lbl, s_lbl)

        # --- Helper: Clean 3b1b Module Block ---
        def make_block(w, h, color, title_tex, math_tex, dashed=False):
            rect = RoundedRectangle(
                corner_radius=0.08,
                width=w,
                height=h,
                fill_color=C_PANEL,
                fill_opacity=0.95,
                stroke_color=color,
                stroke_width=2.2,
            )
            if dashed:
                border = DashedVMobject(
                    RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2.2),
                    num_dashes=32,
                )
                bg = RoundedRectangle(
                    corner_radius=0.08, width=w, height=h,
                    fill_color=C_PANEL, fill_opacity=0.65, stroke_width=0
                )
                rect = VGroup(bg, border)

            t1 = Tex(title_tex, color=color).scale(0.42)
            t2 = MathTex(math_tex, color=WHITE).scale(0.38)
            content = VGroup(t1, t2).arrange(DOWN, buff=0.1).move_to(rect.get_center())
            return VGroup(rect, content)

        # --- Helper: Bottom 3b1b Equation Bar (Guaranteed No Overlap) ---
        hud_frame = RoundedRectangle(
            corner_radius=0.1,
            width=13.4,
            height=1.95,
            fill_color=C_PANEL,
            fill_opacity=0.95,
            stroke_color=C_BORDER,
            stroke_width=2,
        ).to_edge(DOWN, buff=0.18)

        def make_hud(stage_title, eq_top_tex, eq_bot_tex, accent_color):
            title = Tex(stage_title, color=accent_color).scale(0.46)
            eq1 = MathTex(eq_top_tex, color=WHITE).scale(0.48)
            eq2 = MathTex(eq_bot_tex, color=GREY_B).scale(0.44)

            grp = VGroup(title, eq1, eq2).arrange(DOWN, buff=0.14)
            if grp.width > 12.8:
                grp.scale_to_fit_width(12.8)
            grp.move_to(hud_frame.get_center())
            return grp

        # --- HEADER ---
        header = Tex(
            r"\textbf{CrossModalSAROpticalNet} \,---\, Asymmetric Dual-Tower Metric Learning",
            color=WHITE,
        ).scale(0.58).to_edge(UP, buff=0.18)
        header_line = Line(LEFT * 6.7, RIGHT * 6.7, color=C_BORDER, stroke_width=1.5).next_to(header, DOWN, buff=0.1)
        self.add(header, header_line, hud_frame)

        # =========================================================================
        # STAGE 1: MULTI-SENSOR TENSORS & SYNCHRONIZED D4 AUGMENTATION
        # =========================================================================
        sar_stack = make_tensor_stack(
            2, 0.75, C_SAR, r"\textbf{Sentinel-1 SAR}", r"X_{\text{S1}} \in \mathbb{R}^{B \times 2 \times 120 \times 120}"
        ).move_to(LEFT * 5.5 + UP * 1.95)

        opt_stack = make_tensor_stack(
            6, 0.65, C_OPT, r"\textbf{Sentinel-2 MSI}", r"X_{\text{S2}} \in \mathbb{R}^{B \times 10 \times 120 \times 120}"
        ).move_to(LEFT * 5.5 + DOWN * 0.15)

        d4_box = make_block(
            1.75, 2.7, WHITE,
            r"\textbf{Sync } $D_4$",
            r"(T(X_{\text{S1}}),\, T(X_{\text{S2}}))"
        ).move_to(LEFT * 3.35 + UP * 0.9)

        arr_s1_d4 = Arrow(sar_stack[0].get_right(), d4_box.get_left() + UP * 0.85, buff=0.12, color=C_SAR, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        arr_s2_d4 = Arrow(opt_stack[0].get_right(), d4_box.get_left() + DOWN * 0.85, buff=0.12, color=C_OPT, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)

        hud_1 = make_hud(
            r"\textbf{Stage 1: Radiometric Normalization \& Co-Registered Dihedral ($D_4$) Routing}",
            r"\tilde{X}_{\text{S1}} = \frac{\text{clip}(\sigma^0_{\text{dB}}, -35, 0) + 35}{35}, \qquad \tilde{X}_{\text{S2}} = \text{clip}\!\left(\frac{\text{DN}}{10000}, 0, 1\right), \qquad T \in D_4",
            r"\text{Online Query Branch } (\theta_q) \leftarrow \text{Augmented } T(X) \quad\big|\quad \text{Target Key Branch } (\theta_k) \leftarrow \text{Clean Unaugmented } X",
            C_SAR,
        )

        self.play(LaggedStart(FadeIn(sar_stack, shift=RIGHT * 0.2), FadeIn(opt_stack, shift=RIGHT * 0.2), lag_ratio=0.2), FadeIn(hud_1), run_time=1.2)
        self.play(GrowArrow(arr_s1_d4), GrowArrow(arr_s2_d4), FadeIn(d4_box), run_time=1.0)
        # Synchronized 90-degree rotation of the tensor slices inside the stacks
        self.play(
            Rotate(sar_stack[0], angle=PI / 2),
            Rotate(opt_stack[0], angle=PI / 2),
            run_time=1.1,
        )
        self.wait(1.2)

        # =========================================================================
        # STAGE 2: ENERGY-CONSERVING CONVNEXT-TINY STEMS & BACKBONE
        # =========================================================================
        stem_sar = make_block(
            2.15, 0.95, C_SAR,
            r"\textbf{SAR Stem } $4\times 4$",
            r"2 \to 96 \;\; (\text{stride } 4)"
        ).move_to(LEFT * 1.05 + UP * 2.15)

        stem_opt = make_block(
            2.15, 0.95, C_OPT,
            r"\textbf{Opt Stem } $4\times 4$",
            r"10 \to 96 \;\; (\text{stride } 4)"
        ).move_to(LEFT * 1.05 + UP * 0.95)

        arr_d4_st1 = Arrow(d4_box.get_right() + UP * 0.95, stem_sar.get_left(), buff=0.1, color=C_SAR, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)
        arr_d4_st2 = Arrow(d4_box.get_right() + UP * 0.05, stem_opt.get_left(), buff=0.1, color=C_OPT, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)

        hud_2 = make_hud(
            r"\textbf{Stage 2: Energy-Conserving ConvNeXt-Tiny Stem Weight Initialization}",
            r"W_{\text{S1}} = \text{repeat}_2\!\left(\frac{1}{3}\sum_{c=1}^{3} W_{\text{RGB}}[:,c,:,:]\right) \in \mathbb{R}^{96\times 2\times 4\times 4}, \qquad W_{\text{S2}} = \frac{3}{10}\,\text{repeat}_4(W_{\text{RGB}})[:,:10,:,:] \in \mathbb{R}^{96\times 10\times 4\times 4}",
            r"\text{Spatial Hierarchy: } (B, C, 120, 120) \xrightarrow{\text{Stem } 4\times 4} (B, 96, 30, 30) \xrightarrow{\text{Stages 1--4}} F \in \mathbb{R}^{B \times 768 \times 3 \times 3}",
            C_OPT,
        )

        self.play(
            ReplacementTransform(hud_1, hud_2),
            GrowArrow(arr_d4_st1), GrowArrow(arr_d4_st2),
            FadeIn(stem_sar, shift=RIGHT * 0.15),
            FadeIn(stem_opt, shift=RIGHT * 0.15),
            run_time=1.2,
        )
        self.wait(1.8)

        # =========================================================================
        # STAGE 3: ASYMMETRIC ONLINE / TARGET ENCODERS, GEM POOLING & EMA
        # =========================================================================
        online_enc = make_block(
            2.5, 1.35, C_ONLINE,
            r"\textbf{Online Encoder } $(\theta_q)$",
            r"\text{GeM}(p\!=\!3.0) \to \text{MLP} \to \mathbb{S}^{255}"
        ).move_to(RIGHT * 1.75 + UP * 1.85)

        target_enc = make_block(
            2.5, 1.35, C_TARGET,
            r"\textbf{Target Encoder } $(\theta_k)$",
            r"\text{Stop-Grad} \;\big|\; \text{Clean Views}",
            dashed=True,
        ).move_to(RIGHT * 1.75 + DOWN * 0.25)

        arr_st1_on = Arrow(stem_sar.get_right(), online_enc.get_left() + UP * 0.28, buff=0.1, color=C_ONLINE, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)
        arr_st2_on = Arrow(stem_opt.get_right(), online_enc.get_left() + DOWN * 0.28, buff=0.1, color=C_ONLINE, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)

        # Clean unaugmented route from D4 directly to Target Encoder
        arr_clean_tar = Arrow(
            d4_box.get_right() + DOWN * 1.0,
            target_enc.get_left(),
            buff=0.1, color=C_TARGET, stroke_width=2.5, max_tip_length_to_length_ratio=0.12
        )
        clean_lbl = Tex(r"Clean $X_{\text{S1}}, X_{\text{S2}}$ (No Aug)", color=C_TARGET).scale(0.32).next_to(arr_clean_tar, DOWN, buff=0.05)

        ema_arrow = DashedLine(
            online_enc.get_bottom(), target_enc.get_top(),
            color=C_LOSS, stroke_width=2.8
        ).add_tip(tip_length=0.15)
        ema_tex = MathTex(r"\theta_k \leftarrow m\theta_k + (1\!-\!m)\theta_q", color=C_LOSS).scale(0.34).next_to(ema_arrow, RIGHT, buff=0.08)

        hud_3 = make_hud(
            r"\textbf{Stage 3: Learnable GeM Pooling ($p=3.0$), Hypersphere Projection \& EMA Sync}",
            r"g_c = \left(\frac{1}{H'W'}\sum_{u,v} \max(F_{c,u,v}, 10^{-6})^p\right)^{\!1/p} \in \mathbb{R}^{768}, \qquad z = \frac{\phi(g)}{\|\phi(g)\|_2} \in \mathbb{S}^{255}",
            r"\text{EMA Update } (m=0.996): \quad \theta_k \leftarrow 0.996\,\theta_k + 0.004\,\theta_q \quad\text{with synchronized BatchNorm1d } (\mu_{\text{tar}}, \sigma^2_{\text{tar}})",
            C_ONLINE,
        )

        self.play(
            ReplacementTransform(hud_2, hud_3),
            GrowArrow(arr_st1_on), GrowArrow(arr_st2_on),
            GrowArrow(arr_clean_tar), FadeIn(clean_lbl),
            FadeIn(online_enc), FadeIn(target_enc),
            Create(ema_arrow), FadeIn(ema_tex),
            run_time=1.4,
        )
        self.wait(2.0)

        # =========================================================================
        # STAGE 4: HYPERSPHERE S^255 & JACCARD-MASKED INFONCE MATRIX
        # =========================================================================
        iou_gate = make_block(
            2.35, 1.15, C_GEM,
            r"\textbf{CORINE Jaccard Gate}",
            r"\text{IoU}_{ij} = \frac{|Y_i \cap Y_j|}{|Y_i \cup Y_j|} > 0.40"
        ).move_to(RIGHT * 5.15 + UP * 1.95)

        loss_node = make_block(
            2.35, 1.35, C_LOSS,
            r"\textbf{Masked InfoNCE}",
            r"l_{ij} = -\infty \;\text{if } M_{ij}=1"
        ).move_to(RIGHT * 5.15 + DOWN * 0.25)

        arr_on_loss = Arrow(online_enc.get_right(), loss_node.get_left() + UP * 0.28, buff=0.1, color=C_ONLINE, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        arr_tar_loss = Arrow(target_enc.get_right(), loss_node.get_left() + DOWN * 0.28, buff=0.1, color=C_TARGET, stroke_width=2.5, max_tip_length_to_length_ratio=0.18)
        arr_gate_loss = Arrow(iou_gate.get_bottom(), loss_node.get_top(), buff=0.08, color=C_GEM, stroke_width=2.5, max_tip_length_to_length_ratio=0.22)

        hud_4 = make_hud(
            r"\textbf{Stage 4: Multi-Label Jaccard False-Negative Masking \& Composite InfoNCE}",
            r"\mathcal{L}(U,V) = -\frac{1}{2B}\sum_{i=1}^{B} \log \frac{\exp(u_i^\top v_i / \tau)}{\exp(u_i^\top v_i / \tau) + \sum_{j \neq i,\, \text{IoU}_{ij} \le 0.40} \exp(u_i^\top v_j / \tau)} \qquad (\tau = 0.10)",
            r"\mathcal{L}_{\text{total}} = 0.50\,\big[\mathcal{L}(Z_{\text{S1}}, T_{\text{S2}}) + \mathcal{L}(Z_{\text{S2}}, T_{\text{S1}})\big] + 0.25\,\big[\mathcal{L}(Z_{\text{S1}}, T_{\text{S1}}) + \mathcal{L}(Z_{\text{S2}}, T_{\text{S2}})\big]",
            C_LOSS,
        )

        self.play(
            ReplacementTransform(hud_3, hud_4),
            FadeIn(iou_gate, shift=DOWN * 0.15),
            FadeIn(loss_node, shift=UP * 0.15),
            GrowArrow(arr_on_loss),
            GrowArrow(arr_tar_loss),
            GrowArrow(arr_gate_loss),
            run_time=1.4,
        )
        self.play(Indicate(loss_node[0], color=C_LOSS, scale_factor=1.04), run_time=1.0)
        self.wait(3.0)