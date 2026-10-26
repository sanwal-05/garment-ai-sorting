export const CLAIM_OPTIONS = [
  { value: "arrived_stained", label: "Arrived stained" },
  { value: "arrived_damaged", label: "Arrived damaged/torn" },
  { value: "poor_quality", label: "Poor quality / worn fabric" },
  { value: "wrong_size", label: "Wrong size or fit" },
  { value: "not_as_described", label: "Not as described" },
  { value: "changed_mind", label: "Changed my mind" },
] as const;

export type ClaimCategory = (typeof CLAIM_OPTIONS)[number]["value"];

export type CustomerStatus =
  | "received"
  | "approved"
  | "needs_more_info"
  | "under_review"
  | "declined";

export type Decision = "ACCEPT" | "REVIEW" | "REJECT";

export type Disposition =
  | "RESELL_AS_NEW"
  | "RESELL_AS_USED"
  | "REFURBISH"
  | "RECYCLE"
  | "INSPECT_MANUALLY";

export type ScoreKey = "stain_present" | "hole_major" | "heavily_worn";

export type Level = "lo" | "mid" | "hi";

export interface Thresholds {
  lo: number;
  hi: number;
}

export interface AdminReturn {
  id: string;
  created_at: string | null;
  claim_category: ClaimCategory | string;
  claim_text: string;
  front_url: string;
  back_url: string;
  scores: Record<ScoreKey, number>;
  levels: Record<ScoreKey, Level>;
  decision: Decision;
  disposition: Disposition;
  reasons: string[];
  status: CustomerStatus;
  reviewer_override: boolean;
  reviewer_note: string | null;
  model_decision: Decision | null;
  model_disposition: Disposition | null;
  model_status: CustomerStatus | null;
  thresholds: Record<ScoreKey, Thresholds>;
  mock_model?: boolean;
}
