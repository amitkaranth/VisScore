"""
Streamlit UI for csv_tufte_charts: upload CSVs, generate good + bad chart PNGs.

Does not import or modify app.py. Run separately:

  streamlit run streamlit_csv_charts.py
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from csv_tufte_charts import _slug, process_csv

PAGE_TITLE = "CSV → Tufte charts"


def _run_generation(
    uploaded_files: list,
    dpi: int,
    max_charts: int,
    include_bad: bool,
) -> tuple[Path, list[dict]]:
    """Write uploads to a temp dir, run process_csv per file, return (output_root, manifest)."""
    work = Path(tempfile.mkdtemp(prefix="csv_tufte_ui_"))
    out_root = work / "generated_plots"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for i, uf in enumerate(uploaded_files):
        raw = uf.getvalue()
        safe_name = Path(uf.name).name
        in_path = work / f"{i:03d}_{safe_name}"
        in_path.write_bytes(raw)
        stem_prefix = f"{i:03d}_{_slug(Path(safe_name).stem)}"
        try:
            manifest.extend(
                process_csv(
                    in_path,
                    out_root,
                    dpi,
                    max_charts,
                    stem_prefix,
                    include_bad=include_bad,
                )
            )
        except Exception as e:
            manifest.append(
                {
                    "source_csv": str(in_path),
                    "status": "error",
                    "error": str(e),
                    "charts": [],
                }
            )

    manifest_path = out_root / "csv_tufte_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_root, manifest


def _manifest_to_zip_bytes(out_root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_root.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_root))
    buf.seek(0)
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📈", layout="wide")
    st.title(PAGE_TITLE)
    st.caption(
        "Standalone helper for `csv_tufte_charts.py`. Your main VisScore app (`app.py`) is unchanged — run that with `streamlit run app.py`."
    )

    with st.sidebar:
        st.header("Options")
        dpi = st.number_input("DPI", min_value=72, max_value=300, value=150, step=1)
        max_charts = st.number_input(
            "Max chart types per CSV",
            min_value=1,
            max_value=12,
            value=6,
            help="Line, bar, scatter, histogram slots; each can produce good + bad PNGs.",
        )
        include_bad = st.checkbox("Include bad (chartjunk) variants", value=True)
        st.divider()
        st.markdown("**Run**")
        generate = st.button("Generate charts", type="primary", use_container_width=True)

    uploaded = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Column types are inferred automatically (dates, numbers, categories).",
    )

    if "csv_ui_out_root" not in st.session_state:
        st.session_state.csv_ui_out_root = None
        st.session_state.csv_ui_manifest = None
        st.session_state.csv_ui_work = None

    if generate:
        if not uploaded:
            st.warning("Upload at least one CSV first.")
        else:
            with st.spinner("Generating plots…"):
                if st.session_state.csv_ui_work and Path(st.session_state.csv_ui_work).exists():
                    shutil.rmtree(st.session_state.csv_ui_work, ignore_errors=True)
                out_root, manifest = _run_generation(list(uploaded), int(dpi), int(max_charts), include_bad)
                st.session_state.csv_ui_work = str(out_root.parent)
                st.session_state.csv_ui_out_root = str(out_root)
                st.session_state.csv_ui_manifest = manifest
            st.success("Done.")

    out_root_str = st.session_state.csv_ui_out_root
    manifest = st.session_state.csv_ui_manifest

    if not out_root_str or not manifest:
        st.info("Upload CSVs, adjust sidebar options, then click **Generate charts**.")
        return

    out_root = Path(out_root_str)
    manifest_path = out_root / "csv_tufte_manifest.json"
    if manifest_path.is_file():
        st.download_button(
            label="Download manifest (JSON)",
            data=manifest_path.read_bytes(),
            file_name="csv_tufte_manifest.json",
            mime="application/json",
        )
    zip_bytes = _manifest_to_zip_bytes(out_root)
    st.download_button(
        label="Download all PNGs + manifest (ZIP)",
        data=zip_bytes,
        file_name="csv_tufte_charts_export.zip",
        mime="application/zip",
    )

    st.subheader("Results")
    for block in manifest:
        src = block.get("source_csv", "?")
        status = block.get("status", "?")
        charts = block.get("charts", [])
        err = block.get("error")
        with st.expander(f"{Path(src).name} — {status}", expanded=status == "ok"):
            st.text(f"Source: {src}")
            if err:
                st.error(err)
            if not charts:
                st.write("No charts for this file.")
                continue
            good = [c for c in charts if c.get("quality") == "good"]
            bad = [c for c in charts if c.get("quality") == "bad"]
            st.write(f"{len(good)} good, {len(bad)} bad image(s).")
            by_kind: dict[str, dict[str, list]] = {}
            for c in charts:
                k = c.get("kind", "unknown")
                q = c.get("quality", "good")
                by_kind.setdefault(k, {}).setdefault(q, []).append(c)
            for kind in sorted(by_kind.keys()):
                st.markdown(f"**{kind}**")
                row_good = by_kind[kind].get("good", [])
                row_bad = by_kind[kind].get("bad", [])
                if row_bad:
                    cols = st.columns(2)
                    c0, c1 = cols[0], cols[1]
                else:
                    c0, c1 = st.container(), None
                with c0:
                    st.markdown("*Good*")
                    for c in row_good:
                        rel = c.get("file", "")
                        p = out_root / rel
                        if p.is_file():
                            st.image(str(p), caption=rel)
                        else:
                            st.warning(f"Missing: {rel}")
                if c1 is not None:
                    with c1:
                        st.markdown("*Bad*")
                        for c in row_bad:
                            rel = c.get("file", "")
                            p = out_root / rel
                            if p.is_file():
                                viol = c.get("violations") or []
                                cap = rel + (
                                    f" — {', '.join(viol[:3])}…"
                                    if len(viol) > 3
                                    else (f" — {', '.join(viol)}" if viol else "")
                                )
                                st.image(str(p), caption=cap)
                            else:
                                st.warning(f"Missing: {rel}")


if __name__ == "__main__":
    main()
