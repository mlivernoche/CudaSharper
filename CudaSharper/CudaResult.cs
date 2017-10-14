using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaSharper
{
    public struct CudaResult<T> : ICudaResult<T>
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
