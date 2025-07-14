import ast
import re
import pathlib

header = '# API Documentation\n'

# mapping module -> list of objects
modules = {
    'ml.model.builders.cnn': ['build_cnn1d','build_dense_as_conv1d','build_cnn2d','build_dense_as_conv2d','build_cnn3d','build_dense_as_conv3d','build_tcnn1d','build_tcnn2d','build_tcnn3d'],
    'ml.model.builders.dnn': ['build_dnn'],
    'ml.model.builders.gnn': ['build_grid_adjacency','build_knn_adjacency','build_gcn','build_gat','build_cheb'],
    'ml.model.builders.lstm': ['build_lstm'],
    'ml.model.builders.se': ['build_squeeze_excite_1d'],
    'ml.model.builders.skip': ['trial_skip_connections'],
    'ml.model.callbacks': ['get_callbacks_model'],
    'ml.model.stats': ['get_flops','get_macs','get_memory_and_time','get_model_usage_stats','write_model_stats_to_file'],
    'ml.model.tools': ['convert_to_saved_model','punish_model_flops','punish_model_params','punish_model'],
    'ml.model.utils': ['capture_model_summary'],
    'ml.optuna.analyzer': ['analyze_study','PlotConfig','set_plot_config_param','set_plot_config_params'],
    'ml.optuna.callbacks': ['get_callbacks_study','ImprovementStagnation','StopIfKeepBeingPruned','NanLossPrunerOptuna'],
    'ml.optuna.model_tools': ['estimate_training_memory','plot_model_param_distribution','set_user_attr_model_stats'],
    'ml.optuna.utils': ['cleanup_non_top_trials','get_remaining_trials','get_top_trials','rename_top_k_files','save_trial_params_to_file','save_top_k_trials','init_study_dirs'],
    'notifications.email': ['send_email'],
    'runtime.monitoring': ['run_auto_restart'],
    'utils.io': ['create_run_directory'],
    'utils.misc': ['clear','format_number','format_bytes','format_scientific','format_number_commas'],
    'utils.system': ['get_user_gpu_choice','get_gpu_info','gpu_summary','log_resources'],
    'visualization.configs': ['config_plt'],
}

base_path = pathlib.Path('src/araras')

md_parts = [header]

section_cache = {}

for mod, names in modules.items():
    path = base_path.joinpath(*mod.split('.'))
    if path.with_suffix('.py').exists():
        file_path = path.with_suffix('.py')
    else:
        file_path = path / '__init__.py'
    text = file_path.read_text()
    module = ast.parse(text)
    mod_header = f"## {mod}\n"
    md_parts.append(mod_header)

    objects = {name:None for name in names}

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in objects:
            objects[node.name] = node

    for name in names:
        node = objects.get(name)
        if node is None:
            continue
        if isinstance(node, ast.FunctionDef):
            signature_params = [arg.arg for arg in node.args.args]
            signature = name + '(\n' + '\n'.join(f'    {p},' for p in signature_params) + '\n)'
        else:
            signature = name
        doc = ast.get_docstring(node) or ''
        lines = doc.splitlines()
        desc_lines = []
        note_blocks = []
        params = []
        returns = ''
        raises = []
        current = None
        current_tag = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('Args:') or stripped.startswith('Parameters:'):
                current = 'args'
                continue
            elif stripped.startswith('Returns:'):
                current = 'returns'
                returns = stripped[len('Returns:'):].strip()
                continue
            elif stripped.startswith('Raises:'):
                current = 'raises'
                continue
            elif re.match(r'^(Note|Tip|Important|Warning|Caution):', stripped):
                tag, rest = stripped.split(':',1)
                current_tag = tag.upper()
                note_msg = rest.strip()
                note_blocks.append([current_tag, note_msg])
                current = 'note'
                continue

            if current == 'args':
                if stripped == '':
                    continue
                m = re.match(r'(\w+)(?:\s*\(([^)]*)\))?:\s*(.*)', stripped)
                if m:
                    params.append([m.group(1), m.group(2), m.group(3)])
                else:
                    if params:
                        params[-1][2] += ' ' + stripped
            elif current == 'note':
                if line.startswith(' '):
                    note_blocks[-1][1] += ' ' + stripped
                else:
                    current = None
                    if stripped:
                        desc_lines.append(stripped)
            elif current == 'returns':
                if stripped:
                    returns += ' ' + stripped
            elif current == 'raises':
                if stripped:
                    parts = stripped.split('-',1)
                    if len(parts)==2:
                        raises.append(parts[0].strip())
                    else:
                        raises.append(stripped)
            else:
                if stripped:
                    desc_lines.append(stripped)
        description = ' '.join(desc_lines) or 'No description available.'

        # If no parameter info parsed, fall back to annotations
        if isinstance(node, ast.FunctionDef):
            ann_map = {}
            for arg in node.args.args:
                if arg.arg == 'self':
                    continue
                atype = getattr(arg.annotation, 'id', 'Any') if arg.annotation else 'Any'
                ann_map[arg.arg] = atype
            if not params:
                for arg, atype in ann_map.items():
                    params.append([arg, atype, ''])
            else:
                for p in params:
                    if not p[1] and p[0] in ann_map:
                        p[1] = ann_map[p[0]]

        # Determine return annotation if returns not parsed
        if not returns:
            if isinstance(node, ast.FunctionDef) and node.returns:
                if isinstance(node.returns, ast.Name):
                    returns = node.returns.id
                elif isinstance(node.returns, ast.Subscript):
                    returns = 'typing'
                else:
                    returns = 'Any'

        md_parts.append(f"### {name}\n")
        md_parts.append('```python')
        md_parts.append(signature)
        md_parts.append('```')
        md_parts.append(description)
        for tag, msg in note_blocks:
            md_parts.append(f"> [{tag}]\n> {msg}\n")
        md_parts.append('\n**Parameters**')
        md_parts.append('| Name | Type | Description |')
        md_parts.append('|------|------|-------------|')
        for p_name, p_type, p_desc in params:
            md_parts.append(f'| {p_name} | `{p_type}` | {p_desc} |')
        md_parts.append('')
        md_parts.append('**Returns**')
        if returns:
            md_parts.append(f'`{returns}`')
        else:
            md_parts.append('`None`')
        md_parts.append('')
        md_parts.append('**Raises**')
        if raises:
            for r in raises:
                md_parts.append(f'- {r}')
        else:
            md_parts.append('- None')
        md_parts.append('')

out_path = pathlib.Path('docs/API_Documentation.md')
out_path.write_text('\n'.join(md_parts))
print('Wrote', out_path)
