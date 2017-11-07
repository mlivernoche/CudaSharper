namespace CudaSharper
{
    public interface ICudaResult<T>
    {
        CudaError Error { get; }
        T Result { get; }
    }
}
