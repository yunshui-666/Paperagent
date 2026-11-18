"""业务异常契约。status_code 由 HTTP 异常处理器转换为响应，message 用于用户可读反馈。"""

class ProjectError(RuntimeError):
    """携带用户可读说明与 HTTP 状态码的业务异常。"""
    def __init__(self, message, status_code=409):
        """初始化错误说明及状态码，默认按业务冲突处理。"""
        super().__init__(message)
        self.status_code = status_code
