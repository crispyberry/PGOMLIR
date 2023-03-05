; ModuleID = 'better_toy.bc'
source_filename = "better_toy.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z10better_toyiii(i32 noundef %a, i32 noundef %b, i32 noundef %step) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %step.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  store i32 %step, ptr %step.addr, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 214748364
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %a.addr, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %a.addr, align 4
  %2 = load i32, ptr %a.addr, align 4
  %cmp1 = icmp sle i32 %2, 214748164
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %3 = load i32, ptr %b.addr, align 4
  %inc2 = add nsw i32 %3, 1
  store i32 %inc2, ptr %b.addr, align 4
  br label %if.end

if.else:                                          ; preds = %for.body
  store i32 0, ptr %a.addr, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %4 = load i32, ptr %step.addr, align 4
  %5 = load i32, ptr %i, align 4
  %add = add nsw i32 %5, %4
  store i32 %add, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %6 = load i32, ptr %a.addr, align 4
  %7 = load i32, ptr %b.addr, align 4
  %add3 = add nsw i32 %6, %7
  ret i32 %add3
}

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #1 {
entry:
  %retval = alloca i32, align 4
  %step = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 1, ptr %step, align 4
  store i32 214748166, ptr %a, align 4
  store i32 0, ptr %b, align 4
  %0 = load i32, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  %2 = load i32, ptr %step, align 4
  %call = call noundef i32 @_Z10better_toyiii(i32 noundef %0, i32 noundef %1, i32 noundef %2)
  store i32 %call, ptr %x, align 4
  ret i32 0
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git e3ce92b7a904c2b4f0acebdf433121b2e45e5f0c)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
