import { BrainCircuit } from "lucide-react";
import type { ReactNode } from "react";
import type { SessionStatus } from "../types/backend";
import { StatusBadge } from "./StatusBadge";
import { ThemeToggle } from "./ThemeToggle";

type LayoutProps = {
  status: SessionStatus;
  children: ReactNode;
};

export function Layout({ status, children }: LayoutProps) {
  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <BrainCircuit size={24} />
          </div>
          <div>
            <p className="eyebrow">InterviewAI</p>
            <h1>Live Interview Intelligence</h1>
          </div>
        </div>
        <div className="topbar-actions">
          <StatusBadge label="Session" value={status} kind={status} />
          <ThemeToggle />
        </div>
      </header>
      {children}
    </main>
  );
}
