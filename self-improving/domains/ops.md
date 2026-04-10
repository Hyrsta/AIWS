# ops

- When a conda env on this machine is activated from a strict bash wrapper, avoid `set -u` during `source conda.sh` and `conda activate`; some activate scripts assume unset tool variables like `ADDR2LINE` are allowed.
- For destructive cleanup of local agent state, prefer moving the old copy to Trash over permanent deletion.
