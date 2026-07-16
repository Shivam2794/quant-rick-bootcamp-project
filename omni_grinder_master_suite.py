import ast
import os
import sys

TARGET_FILES = [
    # Bootcamp 1 (Block 1-5 & Micro Steps 20-25)
    "momentum_prefilter.py",
    "position_sizer.py",
    "raam_scorer.py",
    "rsi_cross_sectional_engine.py",
    "portfolio_amalgamator.py",
    "carver_pipeline.py",
    "bt2_video21_triple_ema.py",
    "bt2_video22_multifactor_raam.py",
    "bt2_video24_beta_rotations.py",
    "block4_final_blueprint.py",
    "meta_regime_filter.py",
    "data_ingestion.py",
    "backtest_engine.py",
    "advanced_tearsheet.py",
    "main.py",
    # Bootcamp 2 (V1-V13)
    "bt2_video1_carver.py",
    "bt2_video2_donchian.py",
    "bt2_video3_macro_canaries.py",
    "bt2_video4_math_derived.py",
    "bt2_video5_hybrid_sizing.py",
    "bt2_video7_data_pipeline.py",
    "bt2_video9_ensemble_math.py",
    "bt2_video10_cross_sectional_ranking.py",
    "bt2_video12_final_assembly.py",
    "bt2_video13_cross_sectional.py",
    "master_pipeline_v1_to_v13.py"
]

class BrutalASTVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.errors = []
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            # Check for iterrows
            if node.func.attr == 'iterrows':
                self.errors.append(f"Line {node.lineno}: FATAL ERROR - Use of 'iterrows()'. Vectorize immediately.")
            # Check for itertuples
            if node.func.attr == 'itertuples':
                self.errors.append(f"Line {node.lineno}: FATAL ERROR - Use of 'itertuples()'. Vectorize immediately.")
            # Check for negative shift (forward looking)
            if node.func.attr == 'shift':
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        self.errors.append(f"Line {node.lineno}: POTENTIAL LOOKAHEAD - Negative shift detected '.shift(-X)'.")
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                        self.errors.append(f"Line {node.lineno}: POTENTIAL LOOKAHEAD - Negative shift detected '.shift({arg.value})'.")
        self.generic_visit(node)

def run_static_analysis():
    print("="*60)
    print(" PHASE 1: BRUTAL STATIC ANALYSIS (AST) ")
    print("="*60)
    
    total_errors = 0
    
    for f in TARGET_FILES:
        if not os.path.exists(f):
            print(f"[MISSING] {f}")
            continue
            
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"[FAIL] {f} - SYNTAX ERROR: {e}")
            total_errors += 1
            continue
            
        visitor = BrutalASTVisitor(f)
        visitor.visit(tree)
        
        if visitor.errors:
            print(f"[FAIL] {f}:")
            for err in visitor.errors:
                print(f"  -> {err}")
                total_errors += 1
        else:
            print(f"[PASS] {f} - Clean AST.")
            
    return total_errors

if __name__ == "__main__":
    errs = run_static_analysis()
    sys.exit(1 if errs > 0 else 0)
