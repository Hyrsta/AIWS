export type JobStatus = "queued" | "running" | "completed" | "failed" | "terminated";

export interface HealthDefaults {
  ssh_host: string; remote_workdir: string; remote_python: string; sam3d_output_root: string;
  dataset_root: string; cadrille_root: string; cadrille_docker_image: string;
  cadrille_docker_extra_args: string; cadrille_checkpoint: string; cadrille_processor_path: string;
}
export interface SimpleDefaults {
  ssh_host: string; remote_workdir: string; remote_root: string; cadrille_mode: string;
  cadrille_checkpoint_preset: string; cadrille_checkpoint: string; cadrille_n_samples: number; cadrille_batch_size: number;
}
export interface HealthRuntime { docker_ok: boolean; docker_version: string; cadrille_image_present: boolean; cadrille_image: string; }
export interface GpuInfo { index: number; name: string; memory_total_mb: number; memory_free_mb: number; utilization: number; }
export interface Health {
  ok: boolean; workspace_root: string; jobs_root: string;
  defaults: HealthDefaults; simple_defaults: SimpleDefaults; runtime: HealthRuntime; catalog_path: string;
  gpus?: GpuInfo[];
}

export type Vec3 = [number, number, number];
export interface CadrilleInputPoints {
  n_points?: number;
  points: Vec3[];
  backfilled?: boolean;
  provenance?: string | null;
  source_candidate?: string | null;
  source_stem?: string | null;
  generation_id?: number | null;
  mode?: string;
  note?: string;
  [k: string]: unknown;
}
export interface CatalogEntry { bbox_m: Vec3; bbox_mm: Vec3; }
export interface CatalogClass { models: string[]; entries: Record<string, CatalogEntry>; }
export interface Catalog { source: string; classes: Record<string, CatalogClass>; }

export interface JobRequest {
  input_mode?: "image_mask" | "image" | "mesh";
  detect_prompt?: string | null;
  image_filename: string | null; mask_filename: string | null; mesh_filename?: string | null;
  cadrille_checkpoint_preset: "RL" | "SFT"; cadrille_checkpoint: string;
  cadrille_mode: "pc" | "img"; cadrille_mode_label: "PC" | "IMG";
  workpiece_class: string | null; model_code: string | null; postscale_enabled: boolean;
  gpu_index?: number | null;
}
export interface ResultPaths {
  job_root?: string; results_root?: string; sam3d_mesh_glb?: string; sam3d_mesh_stl?: string;
  sam3d_mesh_preview_stl?: string | null;
  sam3d_faces_raw?: number | null; sam3d_faces_kept?: number | null;
  sam3d_verts_raw?: number | null; sam3d_verts_kept?: number | null;
  sam3d_face_budget?: number | null; sam3d_reduce_pct?: number | null;
  cadrille_output_root?: string; selected_mesh?: string; selected_py?: string; selected_brep?: string; cadrille_reselect?: string | null;
  cadrille_input_points?: string | null; cadrille_input_render_grid?: string | null;
  cleaned_brep_step?: string; cleaned_mesh_stl?: string; cleanup_metadata?: string;
  n_bodies_before?: number; n_bodies_after?: number; postscale_dir?: string;
  scaled_mesh_stl?: string; scaled_brep_step?: string; scaled_py?: string; scaled_metadata?: string;
  workpiece_class?: string; model_code?: string; stage_metrics?: string | null; [k: string]: unknown;
}
export interface StageMetrics {
  n_points?: number;
  normalization?: string;
  gt_mesh?: string;
  stages: Record<string, { iou: number | null; cd: number | null; iou_corner?: number | null; cd_corner?: number | null; error?: string }>;
  cadrille_selection_metric?: { mean_iou: number | null; median_cd: number | null; note?: string };
  cadrille_reselect?: {
    best?: string | null;
    selection_protocol?: string;
    candidate_count?: number | null;
    code_valid_count?: number | null;
    code_invalid_count?: number | null;
    boolean_iou_invalid_count?: number | null;
    metric_invalid_count?: number | null;
    selectable_count?: number | null;
  };
}
export interface StageTiming { started_at: number | null; ended_at: number | null; }
export interface JobSummary {
  job_id: string; kind: string; status: JobStatus; ssh_host: string; output_root: string;
  created_at: number; updated_at: number; remote_pid: number | null; exit_code: number | null;
  command: string[]; log_path: string; stage: string; stage_label: string;
  started_at: number | null; ended_at: number | null; stage_timings: Record<string, StageTiming>;
  result_paths: ResultPaths | null; request: JobRequest; error: string | null;
}

export interface Sam3dMetrics { available: boolean; duration_sec?: number; model_init_sec?: number; peak_memory_reserved_mb?: number; peak_memory_allocated_mb?: number; cuda_visible_devices?: string | null; }
export interface CadrilleMetrics { available: boolean; mean_iou?: number; median_cd?: number; invalid_cd?: number; invalid_iou?: number; n_samples?: number; duration_sec?: number; peak_memory_reserved_mb?: number; peak_memory_allocated_mb?: number; device_name?: string; device_total_memory_mb?: number; }
export interface PostscaleMetrics { available: boolean; workpiece_class?: string; model_code?: string; rewrite_mode?: string; canonical_extents?: Vec3; catalog_target_mm?: Vec3; after_scale_mm?: Vec3; max_rel_error?: number; match_ok?: boolean; }
export interface Metrics { job_id: string; sam3d: Sam3dMetrics; cadrille: CadrilleMetrics; postscale: PostscaleMetrics; }

// cadrille_cleanup_metadata.json (fetched via /jobs/{id}/file). Verify keys against a live file in Task 5/Step 5.
export interface CleanupMetadata {
  n_bodies_before: number; n_bodies_after: number; n_bodies_removed: number;
  removed_volume_fraction?: number; confidence_flag: boolean; confidence_reasons: string[]; [k: string]: unknown;
}
interface ReconstructBaseInput {
  cadrille_checkpoint_preset: "RL" | "SFT"; cadrille_mode: "PC" | "IMG";
  workpiece_class?: string | null; model_code?: string | null;
  gpu_index?: number | null;
}

// Reconstruct request payload (client -> POST /jobs/simple-reconstruct)
export type ReconstructInput =
  | (ReconstructBaseInput & { input_mode?: "image_mask"; image: File; mask: File })
  | (ReconstructBaseInput & { input_mode: "image"; image: File; detect_prompt?: string | null })
  | (ReconstructBaseInput & { input_mode: "mesh"; mesh: File });

export interface JobInputs {
  job_id: string;
  input_image: string | null;
  input_mask: string | null;
  input_mesh?: string | null;
}

export interface ScaledMetadata {
  rewrite_mode: string;
  dry_run?: boolean;
  catalog: { workpiece_class: string; model_code: string; bbox_m: Vec3; bbox_mm: Vec3 };
  canonical_bbox: { xlen: number; ylen: number; zlen: number; xmin: number; ymin: number; zmin: number; xmax: number; ymax: number; zmax: number };
  after_scale_bbox_mm: { xlen: number; ylen: number; zlen: number };
  scale: {
    mode: string;
    matrix_3x3: number[][];
    scale_on_catalog_axes?: Record<"X" | "Y" | "Z", number>;
    center_canonical?: number[];
    [k: string]: unknown;
  };
  [k: string]: unknown;
}
