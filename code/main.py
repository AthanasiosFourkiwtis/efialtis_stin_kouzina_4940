# Central entry point for the final stacking pipeline.

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    target = Path(__file__).resolve().parent / "notebooks" / "12_stacking_ensemble.py"
    runpy.run_path(str(target), run_name="__main__")
