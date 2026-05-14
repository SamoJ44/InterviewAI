export type SessionStatus =
  | "idle"
  | "preparing"
  | "calibrating"
  | "active"
  | "paused"
  | "ended";

export type Session = {
  id: string;
  status: SessionStatus;
  started_at?: string | null;
  ended_at?: string | null;
  paused?: boolean;
  pause_count?: number;
  total_paused_seconds?: number;
};

export type Scores = {
  eye_contact: number;
  posture: number;
  stability: number;
  self_touch: number;
  expression?: number | null;
  final: number;
};

export type EmotionProbabilities = {
  angry?: number;
  disgust?: number;
  fear?: number;
  happy?: number;
  neutral?: number;
  sad?: number;
  surprise?: number;
};

export type EmotionResponse = {
  label: string | null;
  confidence: number | null;
  probabilities?: EmotionProbabilities;
  positive_prob?: number;
  tense_prob?: number;
  score?: number | null;
  available: boolean;
  status:
    | "ok"
    | "unavailable"
    | "low_confidence"
    | "model_not_loaded"
    | "face_not_detected"
    | "service_unavailable"
    | "prediction_error"
    | string;
};

export type RawEmotionResponse = {
  label: string | null;
  confidence: number | null;
  probabilities?: EmotionProbabilities;
  available: boolean;
  status: string;
};

export type SessionAverages = {
  eye_contact: number | null;
  posture: number | null;
  stability: number | null;
  self_touch: number | null;
  expression: number | null;
  final: number | null;
};

export type SessionEmotionSummary = {
  dominant_emotion: string | null;
  average_confidence: number | null;
  average_positive_prob: number | null;
  distribution: Record<string, number>;
  average_probabilities?: Record<string, number> | null;
};

export type SessionSummary = {
  frame_count: number;
  valid_expression_frame_count: number;
  valid_emotion_frame_count: number;
  averages: SessionAverages;
  event_counts?: Record<string, number>;
  emotion_summary: SessionEmotionSummary;
  debug?: SessionHistoryDebug;
};

export type SessionHistoryDebug = {
  session_id: string;
  stored_score_frames: number;
  stored_score_samples: Record<string, number>;
  stored_emotion_samples: number;
};

export type Calibration = {
  status?: string;
  active?: boolean;
  ready?: boolean;
  progress?: number;
  countdown_remaining?: number;
  samples?: number;
  message?: string;
};

export type Flags = {
  face_detected?: boolean;
  pose_detected?: boolean;
  hands_detected?: boolean;
  self_touch_active?: boolean;
  eye_status?: string;
  calibration_active?: boolean;
  gaze_away_active?: boolean;
};

export type TimelineItem = {
  id?: string;
  time?: string;
  timestamp?: string;
  severity?: "info" | "warning" | "critical" | string;
  title?: string;
  description?: string;
  category?: string;
  [key: string]: unknown;
};

export type AnalyzeFrameResponse = {
  session_id?: string;
  session?: Session;
  scores?: Scores;
  session_scores?: Partial<Scores> | null;
  session_stats?: any;
  session_debug?: SessionHistoryDebug;
  score_details?: any;
  stability_debug?: any;
  emotion?: EmotionResponse;
  raw_emotion?: RawEmotionResponse;
  calibration?: Calibration;
  flags?: Flags;
  events?: TimelineItem[];
  alerts?: TimelineItem[];
  frame_meta?: {
    width?: number;
    height?: number;
  };
};

export type SessionResponse = Session | { session: Session } | { id: string; status?: SessionStatus };

export type EndSessionResponse = {
  session_id: string;
  status: "ended";
  session: Session;
  summary: SessionSummary;
  recommendations: SessionRecommendations | null;
};

export type RecommendationCategory =
  | "eye_contact"
  | "posture"
  | "stability"
  | "self_touch"
  | "expression"
  | "emotion_presence"
  | "overall";

export type RecommendationPriority = "low" | "medium" | "high";

export type RecommendationArea = {
  category: RecommendationCategory;
  priority: RecommendationPriority;
  message: string;
};

export type SessionRecommendations = {
  overall_assessment: string;
  strengths: string[];
  areas_to_improve: RecommendationArea[];
  next_session_focus: string[];
};
