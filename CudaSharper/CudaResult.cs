namespace CudaSharper
{
    public class CudaResult<T> : ICudaResult<T>
    {
        public CudaError Error { get; }
        public T Result { get; }

        public CudaResult(CudaError error, T result)
        {
            Error = error;
            Result = result;
        }

        public CudaResult(int error, T result)
        {
            Error = DTM.CudaErrorCodes(error);
            Result = result;
        }
    }
}
