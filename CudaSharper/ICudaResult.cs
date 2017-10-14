using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CudaSharper
{
    public interface ICudaResult<T>
    {
        CudaError Error { get; }
        T Result { get; }
    }
}
