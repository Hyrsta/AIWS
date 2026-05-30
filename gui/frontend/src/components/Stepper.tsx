import { cn } from "@/lib/utils";
import { visibleStages, stageToStep } from "@/lib/stages";

interface StepperProps {
  stageLabel: string;
  hasPostscale: boolean;
}

export function Stepper({ stageLabel, hasPostscale }: StepperProps) {
  const stages = visibleStages(hasPostscale);
  const activeStep = stageToStep(stageLabel, hasPostscale);

  return (
    <div className="flex items-center gap-1 overflow-x-auto py-2">
      {stages.map((label, i) => {
        const isDone = i < activeStep;
        const isActive = i === activeStep;
        return (
          <div key={label} className="flex items-center gap-1 min-w-0">
            <div
              data-active={isActive ? "true" : undefined}
              data-done={isDone ? "true" : undefined}
              className={cn(
                "flex flex-col items-center gap-1 rounded-lg border px-3 py-2 text-xs min-w-[120px]",
                isDone && "border-primary/40 bg-primary/10 text-primary",
                isActive && "border-primary bg-primary/20 text-primary font-semibold",
                !isDone && !isActive && "border-border bg-muted/40 text-muted-foreground",
              )}
            >
              <span className="font-mono text-[10px] text-muted-foreground">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="text-center leading-tight">{label}</span>
              {isActive && (
                <span className="h-1 w-full rounded-full bg-primary/40 overflow-hidden">
                  <span className="block h-full w-1/2 bg-primary animate-pulse rounded-full" />
                </span>
              )}
              {isDone && (
                <span className="text-[10px] text-primary">done</span>
              )}
            </div>
            {i < stages.length - 1 && (
              <div
                className={cn(
                  "h-px w-4 flex-none",
                  i < activeStep ? "bg-primary" : "bg-border",
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
