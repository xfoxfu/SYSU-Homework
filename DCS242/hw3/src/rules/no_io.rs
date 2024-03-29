use super::Rule;
use crate::context::Context;
use llvm_ir::{instruction::Call, Constant, Instruction, Name, Operand};

const safe_fns: &[&str] = &[
    "_exit",
    "_Exit",
    "abort",
    "accept",
    "access",
    "aio_error",
    "aio_return",
    "aio_suspend",
    "alarm",
    "bind",
    "cfgetispeed",
    "cfgetospeed",
    "cfsetispeed",
    "cfsetospeed",
    "chdir",
    "chmod",
    "chown",
    "clock_gettime",
    "close",
    "connect",
    "creat",
    "dup",
    "dup2",
    "execl",
    "execle",
    "execv",
    "execve",
    "faccessat",
    "fchdir",
    "fchmod",
    "fchmodat",
    "fchown",
    "fchownat",
    "fcntl",
    "fdatasync",
    "fexecve",
    "fork",
    "fstat",
    "fstatat",
    "fsync",
    "ftruncate",
    "futimens",
    "getegid",
    "geteuid",
    "getgid",
    "getgroups",
    "getpeername",
    "getpgrp",
    "getpid",
    "getppid",
    "getsockname",
    "getsockopt",
    "getuid",
    "kill",
    "link",
    "linkat",
    "listen",
    "lseek",
    "lstat",
    "mkdir",
    "mkdirat",
    "mkfifo",
    "mkfifoat",
    "mknod",
    "mknodat",
    "open",
    "openat",
    "pause",
    "pipe",
    "poll",
    "posix_trace_event",
    "pselect",
    "pthread_kill",
    "pthread_self",
    "pthread_sigmask",
    "raise",
    "read",
    "readlink",
    "readlinkat",
    "recv",
    "recvfrom",
    "recvmsg",
    "rename",
    "renameat",
    "rmdir",
    "select",
    "sem_post",
    "send",
    "sendmsg",
    "sendto",
    "setgid",
    "setpgid",
    "setsid",
    "setsockopt",
    "setuid",
    "shutdown",
    "sigaction",
    "sigaddset",
    "sigdelset",
    "sigemptyset",
    "sigfillset",
    "sigismember",
    "signal",
    "sigpause",
    "sigpending",
    "sigprocmask",
    "sigqueue",
    "sigset",
    "sigsuspend",
    "sleep",
    "sockatmark",
    "socket",
    "socketpair",
    "stat",
    "symlink",
    "symlinkat",
    "tcdrain",
    "tcflow",
    "tcflush",
    "tcgetattr",
    "tcgetpgrp",
    "tcsendbreak",
    "tcsetattr",
    "tcsetpgrp",
    "time",
    "timer_getoverrun",
    "timer_gettime",
    "timer_settime",
    "times",
    "umask",
    "uname",
    "unlink",
    "unlinkat",
    "utime",
    "utimensat",
    "utimes",
    "wait",
    "waitpid",
    "write",
];

pub struct NoIO;

impl Rule for NoIO {
    fn check(f: &Instruction, context: &mut Context) -> bool {
        if let Instruction::Call(Call {
            function:
                either::Either::Right(Operand::ConstantOperand(Constant::GlobalReference {
                    name: Name::Name(name),
                    ..
                })),
            ..
        }) = f
        {
            name != "printf" && name != "scanf"
        } else {
            true
        }
    }
    fn name<'a>() -> &'a str {
        "No I/O Call"
    }
}
