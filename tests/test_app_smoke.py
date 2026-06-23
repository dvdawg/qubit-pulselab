from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    assert not at.exception
    # The app exposes at least one slider and the title.
    assert len(at.slider) > 0
    assert any("Pulse" in m.value for m in at.markdown) or at.title


def test_app_has_helpful_controls():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    # Every slider must carry a help string (the core UX requirement).
    assert all(s.help for s in at.slider)
    assert all(s.help for s in at.selectbox)
    assert all(c.help for c in at.checkbox)
    assert all(b.help for b in at.button)


def test_readout_toggle_renders():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    # The readout checkbox exists and has help; enabling it re-runs cleanly.
    labels = [c.label for c in at.checkbox]
    assert any("measured readout" in lbl.lower() for lbl in labels)
    target = next(c for c in at.checkbox if "measured readout" in c.label.lower())
    target.set_value(True).run()
    assert not at.exception


def test_optimizer_shows_before_after_and_persists():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=180)
    at.run()
    # Use a 2-level model so the DRAG optimization is fast.
    at.slider[0].set_value(2).run()
    at.selectbox[0].set_value("DRAG").run()
    run_btn = next(b for b in at.button if "Run optimizer" in b.label)
    run_btn.click().run()
    assert not at.exception
    # The before/after result section rendered its optimized-fidelity metric...
    assert "Fidelity (optimized)" in [m.label for m in at.metric]
    # ...and it persists across an unrelated rerun (the old app dropped it).
    at.run()
    assert "Fidelity (optimized)" in [m.label for m in at.metric]


def test_crab_optimization_renders_result():
    # CRAB has no scalar params (OptResult.params is None); the result section
    # must still render (and skip the DRAG-only AllXY panel) without error.
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=180)
    at.run()
    at.slider[0].set_value(2).run()  # 2 levels for speed
    at.selectbox[0].set_value("CRAB").run()
    next(b for b in at.button if "Run optimizer" in b.label).click().run()
    assert not at.exception
    assert "Fidelity (optimized)" in [m.label for m in at.metric]


def test_grape_runs_through_control_noise():
    # Previously GRAPE raised on the (non-differentiable) control-noise chain;
    # it should now run via the numerical-gradient fallback and show a result.
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=300)
    at.run()
    at.slider[0].set_value(2).run()  # 2 levels for speed
    cn = next(c for c in at.checkbox if "control noise" in c.label.lower())
    cn.set_value(True).run()
    at.selectbox[0].set_value("GRAPE").run()
    next(b for b in at.button if "Run optimizer" in b.label).click().run()
    assert not at.exception
    assert "Fidelity (optimized)" in [m.label for m in at.metric]
