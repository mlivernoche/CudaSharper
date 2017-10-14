using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/*
 * The starting point of CudaSharper is the CUDA C library called CudaSharperLibrary (CSL).
 * These functions, defined in C, are P/Invoked to the CudaSharper wrapper (CSW).
 * The wrapper is segmented into three responsibilities: the marshaler, the translator,
 * and the exposer.
 * 
 * The marshaler: defined in SafeNativeMethods. These are internal extern static methods.
 * 
 * The translator: DataTranslationMethods (DTM), which is tasked with translating
 * data types and structures between CSL and CSW. This is primarily used for cuBLAS, which
 * uses a column-major in constrast to the row-major notation in C/C++/C#/etc.
 * 
 * Side note: ALL CUDA functions must return ICudaResult<T>. This can be used to convert the int
 * returned by SafeNativeMethods into CudaError. This eliminates the need to duplicate code from
 * SafeNativeMethods to DTM.
 * 
 * The exposer: This is the API that external libraries will call. These methods and classes should be entirely
 * written in C# and should not any idea they are using marshaled data and functions. Any sort of argument validation
 * should be done here as well.
 * 
 * As a diagram, it looks like this:
 * 
 * C# App -> Exposer -> DTM -> Marshaler -> CUDA C -> Marshaler -> DTM -> Exposer -> C# App
 */

namespace CudaSharper
{
    // DSTM = DataTranslationMethods
    internal static class DTM
    {
        internal static CudaError CudaErrorCodes(int error_code)
        {
            if (Enum.IsDefined(typeof(CudaError), error_code))
            {
                return (CudaError)error_code;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Provided CUDA Error code is unknown.");
            }
        }

        internal static T[] FlattenArray<T>(int rows, int columns, T[][] nested_array)
        {
            var flat_array = new T[rows * columns];
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    flat_array[(y * rows) + x] = nested_array[y][x];
                }
            }

            return flat_array;
        }

        internal static T[][] UnflattenArray<T>(int rows, int columns, T[] flat_array)
        {
            var nested_array = new T[rows][];
            for (int y = 0; y < rows; y++)
            {
                nested_array[y] = new T[columns];

                for (int x = 0; x < columns; x++)
                {
                    nested_array[y][x] = flat_array[(y * rows) + x];
                }
            }

            return nested_array;
        }

        #region CudaDevice.cs
        internal static (CudaError Error, string Result) GetCudaDeviceName(int device_id)
        {
            StringBuilder device_name = new StringBuilder(256);
            var error = SafeNativeMethods.GetCudaDeviceName(device_id, device_name);
            return (CudaErrorCodes(error), device_name.ToString());
        }
        #endregion

        #region CudaSettings.cs
        internal static CudaError InitializeCudaContext()
        {
            return CudaErrorCodes(SafeNativeMethods.InitializeCudaContext());
        }
        
        internal static CudaError GetCudaDeviceCount()
        {
            return CudaErrorCodes(SafeNativeMethods.GetCudaDeviceCount());
        }

        internal static CudaError ResetCudaDevice()
        {
            return CudaErrorCodes(SafeNativeMethods.ResetCudaDevice());
        }
        #endregion

        internal static (CudaError Error, float[][] Result) MatrixMultiplyFloat(
            int device_id,
            CUBLAS_OP a_op, CUBLAS_OP b_op,
            float alpha,
            float[][] a,
            float[][] b,
            float beta)
        {
            // .NET does not support marshaling nested arrays between C++ and e.g. C#.
            // If you try, you will get the error message, "There is no marshaling support for nested arrays."
            // Further, the cuBLAS function cublasSgemm/cublasDgemm does not have pointer-to-pointers as arguments (e.g., float**), so we cannot
            // supply a multidimensional array anyway.
            // The solution: flatten arrays so that they can passed to CudaSharperLibrary, and then unflatten whatever it passes back.
            var d_a = FlattenArray(a.Length, a[0].Length, a);
            var d_b = FlattenArray(b.Length, b[0].Length, b);

            // Despite the definition above, this will return the correct size for C. Go figure.
            var d_c = new float[a.Length * b.Length];

            var transa_op = (int)a_op;
            var transb_op = (int)b_op;

            var error = SafeNativeMethods.MatrixMultiplyFloat(
                device_id,
                transa_op, transb_op,
                a.Length, b[0].Length, a[0].Length,
                alpha,
                d_a,
                d_b,
                beta,
                d_c);

            return (CudaErrorCodes(error), UnflattenArray(a.Length, b.Length, d_c));
        }

        internal static (CudaError Error, double[][] Result) MatrixMultiplyDouble(
            int device_id,
            CUBLAS_OP a_op, CUBLAS_OP b_op,
            double alpha,
            double[][] a,
            double[][] b,
            double beta)
        {
            // .NET does not support marshaling nested arrays between C++ and e.g. C#.
            // If you try, you will get the error message, "There is no marshaling support for nested arrays."
            // Further, the cuBLAS function cublasSgemm/cublasDgemm does not have pointer-to-pointers as arguments (e.g., float**), so we cannot
            // supply a multidimensional array anyway.
            // The solution: flatten arrays so that they can passed to CudaSharperLibrary, and then unflatten whatever it passes back.
            var d_a = FlattenArray(a.Length, a[0].Length, a);
            var d_b = FlattenArray(b.Length, b[0].Length, b);

            // Despite the definition above, this will return the correct size for C. Go figure.
            var d_c = new double[a.Length * b.Length];

            var transa_op = (int)a_op;
            var transb_op = (int)b_op;

            var error = SafeNativeMethods.MatrixMultiplyDouble(
                device_id,
                transa_op, transb_op,
                a.Length, b[0].Length, a[0].Length,
                alpha,
                d_a,
                d_b,
                beta,
                d_c);

            return (CudaErrorCodes(error), UnflattenArray(a.Length, b.Length, d_c));
        }
    }
}
