"""Resume original IGD train body with saved model/optimizer states."""
import ast
import inspect
import textwrap

def enable_resume(module,start):
    tree=ast.parse(textwrap.dedent(inspect.getsource(module.train)))
    fn=tree.body[0]
    prefix=[];initial=[];remaining=[];loop_seen=False
    for node in fn.body:
        if isinstance(node,ast.For):loop_seen=True
        if not loop_seen and isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            func=node.value.func
            if isinstance(func,ast.Attribute) and func.attr=='train' and isinstance(func.value,ast.Name) and func.value.id in ['generator','discriminator']:
                continue
        if not loop_seen and isinstance(node,ast.Assign):
            target=node.targets[0]
            if isinstance(target,ast.Name) and target.id=='START_ITER':node.value=ast.Constant(start)
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='generator' and target.attr in ['c','sigma']:
                initial.append(node);continue
            if isinstance(target,ast.Attribute) and target.attr=='requires_grad' and isinstance(target.value,ast.Attribute) and target.value.attr in ['c','sigma']:
                initial.append(node);continue
        remaining.append(node)
    # On resume, saved c/sigma are retained. Everything else, including the
    # original optimizer scheduling and remaining iterations, is unchanged.
    fn.body=remaining
    ast.fix_missing_locations(tree)
    exec(compile(tree,'igd_checkpoint_resume','exec'),module.__dict__)
